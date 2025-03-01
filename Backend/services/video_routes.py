"""
Video processing and synchronization route handler.
"""
import os
import time
import torch
import logging
import tempfile
import numpy as np
from pathlib import Path
from flask import request, Response, stream_with_context
import json
import cv2
import subprocess
import base64
import torchaudio
import sys
import shutil
from typing import Generator, Dict, Any

from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager
from services.resource_monitor import ResourceMonitor
from services.error_handler import ErrorHandler

# Set up more verbose logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize processor with required components."""
        logger.info("Initializing VideoProcessor")
        self.audio_processor = AudioProcessor()
        self.model_manager = ModelManager()
        self.processor, self.model, self.text_model, self.tokenizer = self.model_manager.get_model_components()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Ensure temp directory exists
        self.temp_dir = Path('temp')
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Temp directory: {self.temp_dir}")
        
        # Check Wav2Lip setup
        self._check_wav2lip_setup()

    def _check_wav2lip_setup(self):
        """Verify Wav2Lip installation and required files."""
        logger.info("Checking Wav2Lip setup...")
        wav2lip_dir = Path("Wav2Lip")
        
        if not wav2lip_dir.exists():
            logger.error("Wav2Lip directory not found!")
            return
            
        logger.info(f"Wav2Lip directory found: {wav2lip_dir.absolute()}")
        
        # Check essential files
        files_to_check = [
            ("Wav2Lip/inference.py", "Inference script"),
            ("Wav2Lip/audio.py", "Audio processing module"),
            ("Wav2Lip/face_detection/api.py", "Face detection API"),
            ("Wav2Lip/face_detection/detection/sfd/s3fd.pth", "Face detection model"),
            ("Wav2Lip/checkpoints/wav2lip.pth", "Primary model checkpoint"),
            ("Wav2Lip/checkpoints/wav2lip_gan.pth", "GAN model checkpoint (optional)")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"✓ {description} found: {file_path} ({file_size} bytes)")
            else:
                logger.warning(f"✗ {description} not found: {file_path}")

    def _progress_event(self, progress: int, phase: str) -> str:
        """Generate SSE progress event."""
        logger.debug(f"Progress update: {progress}% - {phase}")
        return f"data: {json.dumps({'progress': progress, 'phase': phase})}\n\n"

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        logger.info(f"Extracting audio from {video_path}")
        audio_path = str(self.temp_dir / 'temp_audio.wav')
        try:
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # Output format
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono
                audio_path
            ]
            logger.info(f"Running FFmpeg command: {' '.join(command)}")
            
            # Run with detailed output
            process = subprocess.run(
                command, 
                capture_output=True,
                text=True,
                check=True
            )
            
            if os.path.exists(audio_path):
                logger.info(f"Audio successfully extracted to {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
            else:
                logger.error(f"Audio extraction failed, file does not exist: {audio_path}")
                
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise ValueError(f"Failed to extract audio from video: {e.stderr}")

    def save_frames(self, video_path: str) -> str:
        """Extract and save video frames."""
        logger.info(f"Saving frames from {video_path}")
        frames_dir = self.temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                raise ValueError("Failed to open video file")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.debug(f"Saved {frame_count} frames so far")
                    
            cap.release()
            logger.info(f"Saved {frame_count} frames to {frames_dir}")
            return str(frames_dir)
        except Exception as e:
            logger.error(f"Failed to save frames: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process video frames: {str(e)}")

    def save_audio_tensor(self, audio_tensor: torch.Tensor, path: str):
        """Save audio tensor to file."""
        logger.info(f"Saving audio tensor of shape {audio_tensor.shape} to {path}")
        try:
            # Ensure audio is mono and properly shaped
            if len(audio_tensor.shape) > 1:
                logger.debug(f"Squeezing audio tensor from shape {audio_tensor.shape}")
                audio_tensor = audio_tensor.squeeze(0)
            
            # Log audio properties
            logger.debug(f"Audio tensor stats: min={audio_tensor.min()}, max={audio_tensor.max()}, mean={audio_tensor.mean()}")
            
            # Convert to numpy and save using torchaudio
            torchaudio.save(
                path,
                audio_tensor.unsqueeze(0),  # Add channel dimension
                sample_rate=16000,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            if os.path.exists(path):
                logger.info(f"Audio saved successfully to {path} (size: {os.path.getsize(path)} bytes)")
            else:
                logger.error(f"Audio save failed, file does not exist: {path}")
        except Exception as e:
            logger.error(f"Failed to save audio tensor: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to save translated audio: {str(e)}")

    def translate_audio(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """Translate audio using SeamlessM4T."""
        logger.info(f"Translating audio of shape {audio.shape} to language {target_language}")
        try:
            # Log available CUDA memory if using GPU
            if self.device.type == 'cuda':
                logger.info(f"CUDA memory before translation: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB total, "
                            f"{torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
                            f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")
            
            # Prepare inputs for model
            logger.debug("Preparing inputs for model")
            inputs = self.processor(
                audios=audio.squeeze().numpy(),
                return_tensors="pt",
                sampling_rate=16000,
                src_lang="eng",  # Explicitly set source language
                tgt_lang=target_language
            )
            
            logger.debug(f"Model inputs prepared, shapes: {', '.join([f'{k}: {v.shape}' for k, v in inputs.items() if hasattr(v, 'shape')])}")
        
            if self.device.type == 'cuda':
                logger.debug("Moving inputs to CUDA")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
            # First generate translated text to verify translation is working
            logger.info(f"Generating text translation in {target_language}")
            with torch.no_grad():
                text_output = self.text_model.generate(
                    input_features=inputs["input_features"],
                    tgt_lang=target_language,
                    num_beams=5,
                    max_new_tokens=200
                )
                text = self.processor.batch_decode(text_output, skip_special_tokens=True)[0]
                logger.info(f"Generated text: {text}")

            # Now generate translated speech
            logger.info(f"Generating translated speech for language {target_language}")
            generation_start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=target_language,
                    num_beams=1,
                    speaker_id=7,
                    return_intermediate_token_ids=False
                )
                
                generation_time = time.time() - generation_start_time
                logger.info(f"Speech generation completed in {generation_time:.2f} seconds")
        
                # Get the waveform from the outputs - handle tuple output
                if isinstance(outputs, tuple):
                    logger.debug(f"Output is a tuple of length {len(outputs)}")
                    translated_audio = outputs[0]  # First element is the waveform
                else:
                    logger.debug(f"Output is a single tensor")
                    translated_audio = outputs
                
                # Ensure we're dealing with a tensor
                if not isinstance(translated_audio, torch.Tensor):
                    logger.error(f"Expected tensor output, got {type(translated_audio)}")
                    raise ValueError(f"Expected tensor output, got {type(translated_audio)}")
            
                # Move to CPU if needed
                if translated_audio.device.type != 'cpu':
                    logger.debug(f"Moving output from {translated_audio.device} to CPU")
                    translated_audio = translated_audio.cpu()
            
                # Normalize audio
                logger.debug(f"Audio before normalization: min={translated_audio.min()}, max={translated_audio.max()}")
                max_val = torch.abs(translated_audio).max()
                if max_val > 0:
                    logger.debug(f"Normalizing audio by dividing by {max_val}")
                    translated_audio = translated_audio / max_val
                    
                logger.debug(f"Audio after normalization: min={translated_audio.min()}, max={translated_audio.max()}")
        
                logger.info(f"Translation successful, audio shape: {translated_audio.shape}")
                return translated_audio
            
        except Exception as e:
            logger.error(f"Audio translation failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to translate audio: {str(e)}")

    def combine_audio_video(self, audio_path: str, video_path: str, output_path: str, delay_ms: int = 540) -> None:
        """Combine audio and video files with precise audio delay."""
        logger.info(f"Combining video {video_path} with audio {audio_path}, delay: {delay_ms}ms")
        try:
            # Verify input files exist
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                raise ValueError(f"Video file does not exist: {video_path}")
                
            if not os.path.exists(audio_path):
                logger.error(f"Audio file does not exist: {audio_path}")
                raise ValueError(f"Audio file does not exist: {audio_path}")
                
            # Check video and audio properties
            video_probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height,duration', 
                 '-of', 'json', video_path],
                capture_output=True,
                text=True
            )
            logger.debug(f"Video properties: {video_probe.stdout}")
            
            audio_probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,sample_rate,channels', 
                 '-of', 'json', audio_path],
                capture_output=True,
                text=True
            )
            logger.debug(f"Audio properties: {audio_probe.stdout}")
            
            command = [
                'ffmpeg', '-y',
                '-i', video_path,      # Input video
                '-i', audio_path,      # Input audio
                '-filter_complex', f'[1:a]adelay={delay_ms}|{delay_ms}[adelayed];[adelayed]aresample=44100[a]',  # Delay and resample audio
                '-map', '0:v',         # Use video from first input
                '-map', '[a]',         # Use delayed audio
                '-c:v', 'copy',        # Copy video codec
                '-c:a', 'aac',         # Convert audio to AAC
                '-shortest',           # Cut to shortest stream
                output_path
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(command)}")
            
            # Run with detailed output
            process = subprocess.run(
                command, 
                capture_output=True,
                text=True,
                check=True
            )
            
            if os.path.exists(output_path):
                logger.info(f"Combined video saved to {output_path} (size: {os.path.getsize(output_path)} bytes)")
            else:
                logger.error(f"Combined video not found at expected path: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to combine audio and video: {str(e)}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise ValueError(f"Failed to create final video: {e.stderr}")

    def add_silence_padding(self, audio_tensor: torch.Tensor, silence_duration_samples: int) -> torch.Tensor:
        """Add silence padding to the start of the audio."""
        logger.info(f"Adding {silence_duration_samples} samples of silence padding")
        padding = torch.zeros(1, silence_duration_samples)
        result = torch.cat([padding, audio_tensor], dim=1)
        logger.debug(f"Original shape: {audio_tensor.shape}, padded shape: {result.shape}")
        return result

    def detect_speech_start(self, audio_path: str) -> float:
        """Detect when speech starts in the audio file."""
        logger.info(f"Detecting speech start in {audio_path}")
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file does not exist: {audio_path}")
                return 0.0
                
            waveform, sr = torchaudio.load(audio_path)
            logger.debug(f"Loaded audio with shape {waveform.shape}, sample rate {sr}")
        
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                logger.debug("Converting stereo to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        
            # Calculate RMS energy
            frame_length = int(0.02 * sr)  # 20ms frames
            hop_length = frame_length // 2  # 50% overlap
            
            logger.debug(f"Using frame length {frame_length} and hop length {hop_length}")
        
            frames = waveform.unfold(1, frame_length, hop_length)
            rms = torch.sqrt(torch.mean(frames ** 2, dim=2))
            
            logger.debug(f"Calculated RMS energy for {rms.shape[1]} frames")
        
            # Find first frame above threshold
            mean_energy = torch.mean(rms).item()
            threshold = mean_energy * 0.1  # 10% of mean energy
            logger.debug(f"Mean energy: {mean_energy:.6f}, threshold: {threshold:.6f}")
            
            # Find indices where RMS is above threshold
            above_threshold = torch.where(rms > threshold)[1]
            
            if len(above_threshold) == 0:
                logger.warning("No frames above threshold found, defaulting to start")
                return 0.0
                
            speech_start_frame = above_threshold[0].item()
            logger.debug(f"First frame above threshold: {speech_start_frame}")
        
            # Convert frame index to seconds
            speech_start_time = (speech_start_frame * hop_length) / sr
            logger.info(f"Detected speech start at {speech_start_time:.3f} seconds")
        
            return speech_start_time
        except Exception as e:
            logger.error(f"Failed to detect speech start: {str(e)}", exc_info=True)
            return 0.0  # Default to start if detection fails

    def detect_face(self, video_path):
        """Detect main face in video frames using Wav2Lip's face detection."""
        logger.info(f"Detecting face in {video_path}")
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return None
                
            # Check video properties
            cap_check = cv2.VideoCapture(video_path)
            if not cap_check.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return None
                
            width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_check.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_check.release()
            
            logger.debug(f"Video properties: {width}x{height}, {fps}fps, {frame_count} frames")
            
            # Use face detection from Wav2Lip
            logger.debug("Importing Wav2Lip face detection module")
            sys.path.append('Wav2Lip')
            from face_detection import api as face_detection
        
            # Initialize face detector - EXPLICITLY use CPU
            logger.debug("Initializing face detector")
            detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                device='cpu',  # Force CPU usage
                flip_input=False
            )

            # Extract face from first frame
            logger.debug("Reading first frame for face detection")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read video frame")
                cap.release()
                return None
                
            logger.debug(f"Read frame with shape {frame.shape}")
        
            # Get face detection
            logger.debug("Converting frame to RGB for face detection")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            logger.debug("Running face detection")
            preds = detector.get_detections_for_batch(np.array([frame_rgb]))
            cap.release()
        
            if preds is None:
                logger.warning("Face detection returned None")
                return None
                
            if len(preds) == 0:
                logger.warning("No face detected in video")
                return None
                
            # Log all faces detected
            logger.debug(f"Detected {len(preds)} faces")
            for i, pred in enumerate(preds):
                logger.debug(f"Face {i+1}: {pred}")
        
            # Return bounding box of first face
            logger.info(f"Selected face detected with bounding box: {preds[0]}")
            return preds[0]
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}", exc_info=True)
            return None

    def apply_lip_sync(self, video_path, audio_path, output_path):
        """Apply lip sync using Wav2Lip with error handling."""
        try:
            logger.info(f"Starting lip sync with Wav2Lip: {video_path} + {audio_path}")
            
            # Check if input files exist and have content
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return False
                
            if not os.path.exists(audio_path):
                logger.error(f"Audio file does not exist: {audio_path}")
                return False
                
            video_size = os.path.getsize(video_path)
            audio_size = os.path.getsize(audio_path)
            logger.debug(f"Input file sizes: video={video_size} bytes, audio={audio_size} bytes")
            
            if video_size == 0:
                logger.error("Video file is empty")
                return False
                
            if audio_size == 0:
                logger.error("Audio file is empty")
                return False
            
            # Ensure Wav2Lip directory exists
            wav2lip_dir = Path("Wav2Lip")
            if not wav2lip_dir.exists():
                logger.error("Wav2Lip directory not found. Please clone the Wav2Lip repository.")
                return False
                
            # Check if inference.py exists and is executable
            inference_path = wav2lip_dir / "inference.py"
            if not inference_path.exists():
                logger.error(f"Wav2Lip inference script not found at {inference_path}")
                return False
                
            logger.debug(f"Found Wav2Lip inference script: {inference_path}")
                
            # Check if checkpoint exists - try both regular and GAN models
            checkpoint_path = "Wav2Lip/checkpoints/wav2lip.pth"
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Regular model not found at {checkpoint_path}, trying GAN model")
                checkpoint_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
                if not os.path.exists(checkpoint_path):
                    logger.error("Wav2Lip model checkpoint not found. Please download it to Wav2Lip/checkpoints/.")
                    return False
                    
            checkpoint_size = os.path.getsize(checkpoint_path)
            logger.debug(f"Using checkpoint: {checkpoint_path} (size: {checkpoint_size} bytes)")
                    
            # Ensure face detector model is available
            face_detector_path = "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
            if not os.path.exists(face_detector_path):
                logger.warning(f"Face detector model not found at expected path: {face_detector_path}")
            else:
                face_detector_size = os.path.getsize(face_detector_path)
                logger.debug(f"Face detector model found: {face_detector_path} (size: {face_detector_size} bytes)")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                logger.debug(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            # Check if Python environment has the required packages
            try:
                logger.debug("Importing required Wav2Lip modules to verify installation")
                sys.path.append('Wav2Lip')
                import Wav2Lip.audio
                import Wav2Lip.face_detection
                import Wav2Lip.models
                logger.debug("Successfully imported Wav2Lip modules")
            except ImportError as e:
                logger.error(f"Failed to import Wav2Lip modules: {str(e)}")
                logger.error("Make sure all requirements are installed")
                return False
            
            # Set environment variables to control execution
            execution_env = {
                **os.environ, 
                "PYTHONPATH": f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}",
                "PYTHONUNBUFFERED": "1"  # This ensures Python output isn't buffered
            }
            
            logger.debug(f"Environment PYTHONPATH: {execution_env['PYTHONPATH']}")
            
            # Create command for Wav2Lip with additional parameters for troubleshooting
            command = [
                "python", "Wav2Lip/inference.py",
                "--checkpoint_path", checkpoint_path,
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--nosmooth",              
                "--pads", "0", "20", "0", "0",
                "--resize_factor", "2"  # Using known working value
            ]
            
            logger.info(f"Running Wav2Lip with command: {' '.join(command)}")
            
            # Run Wav2Lip as a subprocess with proper error handling
            process = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                env=execution_env
            )
            
            # Log complete stdout and stderr
            if process.stdout:
                logger.info(f"Process stdout: {process.stdout}")
            else:
                logger.debug("Process stdout was empty")
                
            if process.stderr:
                logger.error(f"Process stderr: {process.stderr}")
            else:
                logger.debug("Process stderr was empty")
            
            # Check for errors
            if process.returncode != 0:
                logger.error(f"Wav2Lip process failed with return code {process.returncode}")
                
                # Try with GAN model if the regular model failed and we're not already using it
                if "wav2lip.pth" in checkpoint_path:
                    gan_checkpoint = "Wav2Lip/checkpoints/wav2lip_gan.pth"
                    if os.path.exists(gan_checkpoint):
                        logger.info("Trying with GAN model instead...")
                        alt_command = command.copy()
                        # Replace checkpoint path
                        checkpoint_index = alt_command.index("--checkpoint_path") + 1
                        alt_command[checkpoint_index] = gan_checkpoint
                        
                        logger.info(f"Running Wav2Lip with GAN model: {' '.join(alt_command)}")
                        alt_process = subprocess.run(
                            alt_command,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=execution_env
                        )
                        
                        # Log complete stdout and stderr from GAN attempt
                        if alt_process.stdout:
                            logger.info(f"GAN model stdout: {alt_process.stdout}")
                        if alt_process.stderr:
                            logger.error(f"GAN model stderr: {alt_process.stderr}")
                        
                        if alt_process.returncode == 0:
                            logger.info("Wav2Lip successful with GAN model!")
                            # Verify output file
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                logger.info(f"Output file created successfully: {output_path} (size: {os.path.getsize(output_path)} bytes)")
                                return True
                            else:
                                logger.error(f"Output file missing or empty: {output_path}")
                                return False
                        else:
                            logger.error(f"GAN model also failed with return code {alt_process.returncode}")
                
                # Try with different padding if both models failed or if GAN model wasn't available
                logger.info("Trying with different padding values...")
                alt_command = command.copy()
                # Replace padding values with different ones
                pad_index = alt_command.index("--pads") + 1
                alt_command[pad_index:pad_index+4] = ["10", "10", "10", "10"]
                
                # Also try a different resize factor
                resize_index = alt_command.index("--resize_factor") + 1
                alt_command[resize_index] = "2"
                
                logger.info(f"Running Wav2Lip with alternative parameters: {' '.join(alt_command)}")
                alt_process = subprocess.run(
                    alt_command,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=execution_env
                )
                
                # Log complete stdout and stderr from alternative parameters attempt
                if alt_process.stdout:
                    logger.info(f"Alt parameters stdout: {alt_process.stdout}")
                if alt_process.stderr:
                    logger.error(f"Alt parameters stderr: {alt_process.stderr}")
                
                if alt_process.returncode == 0:
                    logger.info("Wav2Lip successful with alternative parameters!")
                    # Verify output file
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"Output file created successfully: {output_path} (size: {os.path.getsize(output_path)} bytes)")
                        return True
                    else:
                        logger.error(f"Output file missing or empty: {output_path}")
                        return False
                else:
                    logger.error(f"Alternative parameters also failed with return code {alt_process.returncode}")
                
                # If all attempts failed, check if it's a librosa version issue
                if "TypeError: mel() takes 0 positional arguments but" in process.stderr:
                    logger.error("Detected librosa version incompatibility. This is a known issue.")
                    logger.error("Please edit Wav2Lip/audio.py to change librosa.filters.mel() calls to use keyword arguments")
                    logger.error("Find the _build_mel_basis function and update it to use 'sr=' and 'n_fft=' instead of positional arguments")
                
                return False
            
            # Success case
            logger.info(f"Wav2Lip process completed successfully. Output saved to {output_path}")
            
            # Verify output file exists and has non-zero size
            if not os.path.exists(output_path):
                logger.error(f"Output file is missing: {output_path}")
                return False
                
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                logger.error(f"Output file is empty: {output_path}")
                return False
                
            logger.info(f"Output file verified: {output_path} (size: {output_size} bytes)")
            
            # Verify the output file is a valid video
            try:
                logger.debug("Verifying output video file integrity")
                verification = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 
                     'stream=codec_type,width,height,duration', 
                     '-of', 'json', output_path],
                    capture_output=True,
                    text=True
                )
                logger.debug(f"Output video properties: {verification.stdout}")
                
                if verification.returncode != 0 or verification.stderr:
                    logger.warning(f"Output video verification showed issues: {verification.stderr}")
                    if "Invalid data found when processing input" in verification.stderr:
                        logger.error("Output file appears to be corrupt")
                        return False
            except Exception as e:
                logger.warning(f"Error during output verification: {str(e)}")
                
            return True
                
        except subprocess.TimeoutExpired:
            logger.error("Wav2Lip process timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Lip sync with Wav2Lip failed: {str(e)}", exc_info=True)
            return False

    def process_video(self, video_file, target_language: str) -> Generator[str, None, None]:
        """Process video file and yield progress events."""
        start_time = time.time()
        logger.info(f"Starting video processing for language: {target_language}")
        
        temp_files = []
        try:
            # Log input video file info
            logger.info(f"Input video filename: {video_file.filename}, content type: {video_file.content_type}")
            
            # Save uploaded video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                logger.debug(f"Saving uploaded video to temporary file: {temp_video.name}")
                video_file.save(temp_video.name)
                temp_files.append(temp_video.name)
                
                video_size = os.path.getsize(temp_video.name)
                logger.info(f"Saved uploaded video (size: {video_size} bytes)")
            
            yield self._progress_event(10, "Extracting audio...")
        
            # Extract audio
            logger.debug("Extracting audio from video")
            audio_path = self.extract_audio(temp_video.name)
            temp_files.append(audio_path)
            
            audio_size = os.path.getsize(audio_path)
            logger.debug(f"Extracted audio size: {audio_size} bytes")
        
            # Detect speech start time
            logger.debug("Detecting speech start time")
            speech_start_time = self.detect_speech_start(audio_path)
            logger.info(f"Detected speech starts at: {speech_start_time:.2f} seconds")
        
            yield self._progress_event(20, "Processing audio for translation...")
        
            # Process audio
            logger.debug("Processing audio file for translation")
            audio_tensor = self.audio_processor.process_audio(audio_path)
            logger.debug(f"Processed audio tensor shape: {audio_tensor.shape}")
        
            yield self._progress_event(40, "Translating speech...")
        
            # Translate audio
            logger.debug(f"Translating audio to {target_language}")
            translation_start = time.time()
            translated_audio = self.translate_audio(audio_tensor, target_language)
            translation_time = time.time() - translation_start
            logger.info(f"Translation completed in {translation_time:.2f} seconds")
        
            # Save translated audio
            translated_audio_path = str(self.temp_dir / 'translated_audio.wav')
            logger.debug(f"Saving translated audio to {translated_audio_path}")
            self.save_audio_tensor(translated_audio, translated_audio_path)
            temp_files.append(translated_audio_path)
            
            translated_audio_size = os.path.getsize(translated_audio_path)
            logger.debug(f"Translated audio size: {translated_audio_size} bytes")
        
            yield self._progress_event(60, "Detecting faces for lip synchronization...")
        
            # Check if faces can be detected
            logger.debug("Detecting faces in video")
            face_detection_start = time.time()
            face_bbox = self.detect_face(temp_video.name)
            face_detection_time = time.time() - face_detection_start
            
            has_face = face_bbox is not None
            logger.info(f"Face detection completed in {face_detection_time:.2f} seconds, result: {has_face}")
            
            if has_face:
                logger.debug(f"Face detected with bounding box: {face_bbox}")
            else:
                logger.warning("No face detected in video, will use basic audio sync")
        
            yield self._progress_event(70, "Synchronizing lip movements...")
        
            # Path for the synchronized video
            synced_video = self.temp_dir / 'synced_video.mp4'
            logger.debug(f"Output path for synchronized video: {synced_video}")
        
            if has_face:
                # Apply lip sync using Wav2Lip
                logger.info("Attempting lip sync with Wav2Lip")
                lip_sync_start = time.time()
                success = self.apply_lip_sync(
                    temp_video.name,  # Original video
                    translated_audio_path,  # Translated audio
                    str(synced_video)  # Output path
                )
                lip_sync_time = time.time() - lip_sync_start
                logger.info(f"Lip sync attempt completed in {lip_sync_time:.2f} seconds, success: {success}")
                
                if not success:
                    # Fall back to simpler method if lip sync fails
                    logger.warning("Wav2Lip sync failed, falling back to basic audio sync")
                    delay_ms = int(speech_start_time * 1000)  # Convert to milliseconds
                    logger.debug(f"Using audio delay of {delay_ms}ms for basic sync")
                    
                    basic_sync_start = time.time()
                    self.combine_audio_video(
                        translated_audio_path, 
                        temp_video.name, 
                        str(synced_video),
                        delay_ms
                    )
                    basic_sync_time = time.time() - basic_sync_start
                    logger.info(f"Basic audio sync completed in {basic_sync_time:.2f} seconds")
            else:
                # No face detected, use simple audio delay method
                logger.info("Using basic audio sync (no face detected)")
                delay_ms = int(speech_start_time * 1000)  # Convert to milliseconds
                logger.debug(f"Using audio delay of {delay_ms}ms")
                
                basic_sync_start = time.time()
                self.combine_audio_video(
                    translated_audio_path, 
                    temp_video.name, 
                    str(synced_video),
                    delay_ms
                )
                basic_sync_time = time.time() - basic_sync_start
                logger.info(f"Basic audio sync completed in {basic_sync_time:.2f} seconds")
        
            temp_files.append(str(synced_video))
            
            # Verify final video
            if os.path.exists(str(synced_video)):
                final_size = os.path.getsize(str(synced_video))
                logger.info(f"Final video created: {synced_video} (size: {final_size} bytes)")
            else:
                logger.error(f"Final video not found at expected path: {synced_video}")
        
            yield self._progress_event(90, "Finalizing video...")
        
            # Return the final video data
            logger.debug("Reading final video file for response")
            encoding_start = time.time()
            with open(synced_video, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            encoding_time = time.time() - encoding_start
            
            data_length = len(video_data)
            logger.info(f"Encoded video data in {encoding_time:.2f} seconds (length: {data_length} characters)")
            logger.debug(f"Sending final result to client")
        
            yield f"data: {json.dumps({'result': video_data})}\n\n"
            
            total_time = time.time() - start_time
            logger.info(f"Total video processing completed in {total_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        finally:
            # Cleanup temp files
            logger.debug(f"Cleaning up {len(temp_files)} temporary files")
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            logger.debug(f"Removing directory: {file_path}")
                            shutil.rmtree(file_path)
                        else:
                            logger.debug(f"Removing file: {file_path}")
                            os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Failed to cleanup {file_path}: {str(e)}")
            logger.debug("Temporary file cleanup complete")

def handle_video_processing(target_language: str):
    """Route handler for video processing requests."""
    logger.info(f"Received video processing request for language: {target_language}")
    
    if 'video' not in request.files:
        logger.error("No video file provided in request")
        return ErrorHandler.format_validation_error('No video file provided')
        
    video_file = request.files['video']
    if not video_file.filename:
        logger.error("Empty video filename in request")
        return ErrorHandler.format_validation_error('No video file selected')
    
    logger.info(f"Processing video: {video_file.filename} for language: {target_language}")
    
    processor = VideoProcessor()
    return Response(
        stream_with_context(processor.process_video(video_file, target_language)),
        mimetype='text/event-stream'
    )