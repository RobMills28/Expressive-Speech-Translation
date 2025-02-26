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
from typing import Generator, Dict, Any

from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager
from services.resource_monitor import ResourceMonitor
from services.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize processor with required components."""
        self.audio_processor = AudioProcessor()
        self.model_manager = ModelManager()
        self.processor, self.model, self.text_model, self.tokenizer = self.model_manager.get_model_components()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure temp directory exists
        self.temp_dir = Path('temp')
        self.temp_dir.mkdir(exist_ok=True)

    def _progress_event(self, progress: int, phase: str) -> str:
        """Generate SSE progress event."""
        return f"data: {json.dumps({'progress': progress, 'phase': phase})}\n\n"

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
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
            subprocess.run(command, check=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            raise ValueError("Failed to extract audio from video")

    def save_frames(self, video_path: str) -> str:
        """Extract and save video frames."""
        frames_dir = self.temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
            cap.release()
            return str(frames_dir)
        except Exception as e:
            logger.error(f"Failed to save frames: {str(e)}")
            raise ValueError("Failed to process video frames")

    def save_audio_tensor(self, audio_tensor: torch.Tensor, path: str):
        """Save audio tensor to file."""
        try:
            # Ensure audio is mono and properly shaped
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze(0)
            
            # Convert to numpy and save using torchaudio
            torchaudio.save(
                path,
                audio_tensor.unsqueeze(0),  # Add channel dimension
                sample_rate=16000,
                encoding='PCM_S',
                bits_per_sample=16
            )
        except Exception as e:
            logger.error(f"Failed to save audio tensor: {str(e)}")
            raise ValueError("Failed to save translated audio")

    def translate_audio(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """Translate audio using SeamlessM4T."""
        try:
            # Prepare inputs for model
            inputs = self.processor(
                audios=audio.squeeze().numpy(),
                return_tensors="pt",
                sampling_rate=16000,
                src_lang="eng",  # Explicitly set source language
                tgt_lang=target_language
            )
        
            if self.device.type == 'cuda':
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
            logger.info("Generating translated speech")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=target_language,
                    num_beams=1,
                    speaker_id=7,
                    return_intermediate_token_ids=False
                )
        
                # Get the waveform from the outputs - handle tuple output
                if isinstance(outputs, tuple):
                    translated_audio = outputs[0]  # First element is the waveform
                else:
                    translated_audio = outputs
                
                # Ensure we're dealing with a tensor
                if not isinstance(translated_audio, torch.Tensor):
                    raise ValueError(f"Expected tensor output, got {type(translated_audio)}")
            
                # Move to CPU if needed
                translated_audio = translated_audio.cpu()
            
                # Normalize audio
                max_val = torch.abs(translated_audio).max()
                if max_val > 0:
                    translated_audio = translated_audio / max_val
        
                logger.info(f"Translation successful, audio shape: {translated_audio.shape}")
                return translated_audio
            
        except Exception as e:
            logger.error(f"Audio translation failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to translate audio: {str(e)}")

    def combine_audio_video(self, audio_path: str, video_path: str, output_path: str, delay_ms: int = 540) -> None:
        """Combine audio and video files with precise audio delay."""
        try:
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
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to combine audio and video: {str(e)}")
            raise ValueError("Failed to create final video")

    def add_silence_padding(self, audio_tensor: torch.Tensor, silence_duration_samples: int) -> torch.Tensor:
        """Add silence padding to the start of the audio."""
        padding = torch.zeros(1, silence_duration_samples)
        return torch.cat([padding, audio_tensor], dim=1)

    def detect_speech_start(self, audio_path: str) -> float:
        """Detect when speech starts in the audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        
            # Calculate RMS energy
            frame_length = int(0.02 * sr)  # 20ms frames
            hop_length = frame_length // 2  # 50% overlap
        
            frames = waveform.unfold(1, frame_length, hop_length)
            rms = torch.sqrt(torch.mean(frames ** 2, dim=2))
        
            # Find first frame above threshold
            threshold = torch.mean(rms) * 0.1  # 10% of mean energy
            speech_start_frame = torch.where(rms > threshold)[1][0]
        
            # Convert frame index to seconds
            speech_start_time = (speech_start_frame * hop_length) / sr
        
            return speech_start_time
        except Exception as e:
            logger.error(f"Failed to detect speech start: {str(e)}")
            return 0.0  # Default to start if detection fails

    def detect_face(self, video_path):
        """Detect main face in video frames."""
        try:
            # Import face detection from Diff2Lip
            import sys
            sys.path.append('diff2lip')
            from face_detection import api as face_detection
            
            # Initialize face detector - EXPLICITLY use CPU
            detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                device='cpu',  # Force CPU usage
                flip_input=False
            )

            # Extract face from first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read video frame")
                return None
            
            # Get face detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = detector.get_detections_for_batch(np.array([frame_rgb]))
            cap.release()
            
            if preds is None or len(preds) == 0:
                logger.warning("No face detected in video")
                return None
            
            # Return bounding box
            logger.info(f"Face detected with bounding box: {preds[0]}")
            return preds[0]
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}", exc_info=True)
            return None

    def apply_lip_sync(self, video_path, audio_path, output_path):
        """Apply lip sync using Diff2Lip."""
        try:
            model_path = "diff2lip/checkpoints/archive"
            
            # Updated command with correct argument names
            command = [
                "python", "diff2lip/generate.py",
                "--model_path", model_path,
                "--video_path", video_path,  # Changed from --face
                "--audio_path", audio_path,  # Changed from --audio
                "--out_path", output_path,   # Changed from --outfile
                "--sampling_ref_type", "gt"  # Use ground truth frames as reference
            ]
            
            logger.info(f"Running Diff2Lip with command: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Diff2Lip failed: {process.stderr}")
                return False
                
            logger.info(f"Lip sync successful, output saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Lip sync failed: {str(e)}", exc_info=True)
            return False

    def process_video(self, video_file, target_language: str) -> Generator[str, None, None]:
        """Process video file and yield progress events."""
        temp_files = []
        try:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                video_file.save(temp_video.name)
                temp_files.append(temp_video.name)
            
            yield self._progress_event(10, "Extracting audio...")
        
            # Extract audio
            audio_path = self.extract_audio(temp_video.name)
            temp_files.append(audio_path)
        
            # Detect speech start time
            speech_start_time = self.detect_speech_start(audio_path)
            logger.info(f"Detected speech starts at: {speech_start_time:.2f} seconds")
        
            yield self._progress_event(20, "Processing audio for translation...")
        
            # Process audio
            audio_tensor = self.audio_processor.process_audio(audio_path)
        
            yield self._progress_event(40, "Translating speech...")
        
            # Translate audio
            translated_audio = self.translate_audio(audio_tensor, target_language)
        
            # Save translated audio without padding (we'll use adelay instead)
            translated_audio_path = str(self.temp_dir / 'translated_audio.wav')
            self.save_audio_tensor(translated_audio, translated_audio_path)
            temp_files.append(translated_audio_path)
        
            yield self._progress_event(60, "Detecting faces for lip synchronization...")
        
            # Check if faces can be detected
            has_face = self.detect_face(temp_video.name) is not None
        
            yield self._progress_event(70, "Synchronizing lip movements...")
        
            # Path for the synchronized video
            synced_video = self.temp_dir / 'synced_video.mp4'
        
            if has_face:
                # Apply lip sync using Diff2Lip
                success = self.apply_lip_sync(
                    temp_video.name,  # Original video
                    translated_audio_path,  # Translated audio
                    str(synced_video)  # Output path
                )
                
                if not success:
                    # Fall back to simpler method if lip sync fails
                    logger.warning("Diff2Lip sync failed, falling back to basic audio sync")
                    delay_ms = int(speech_start_time * 1000)  # Convert to milliseconds
                    self.combine_audio_video(
                        translated_audio_path, 
                        temp_video.name, 
                        str(synced_video),
                        delay_ms
                    )
            else:
                # No face detected, use simple audio delay method
                logger.warning("No face detected, using basic audio sync")
                delay_ms = int(speech_start_time * 1000)  # Convert to milliseconds
                self.combine_audio_video(
                    translated_audio_path, 
                    temp_video.name, 
                    str(synced_video),
                    delay_ms
                )
        
            temp_files.append(str(synced_video))
        
            yield self._progress_event(90, "Finalizing video...")
        
            # Return the final video data
            with open(synced_video, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
        
            yield f"data: {json.dumps({'result': video_data})}\n\n"
        
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        finally:
            # Cleanup temp files
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                        else:
                            os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Failed to cleanup {file_path}: {str(e)}")

def handle_video_processing(target_language: str):
    """Route handler for video processing requests."""
    if 'video' not in request.files:
        return ErrorHandler.format_validation_error('No video file provided')
        
    video_file = request.files['video']
    if not video_file.filename:
        return ErrorHandler.format_validation_error('No video file selected')
        
    processor = VideoProcessor()
    return Response(
        stream_with_context(processor.process_video(video_file, target_language)),
        mimetype='text/event-stream'
    )