# services/video_routes.py
import os
import time
import torch
import logging
import tempfile
import numpy as np
from pathlib import Path
from flask import request, Response, stream_with_context, jsonify # Added jsonify
from werkzeug.utils import secure_filename # <--- ADD THIS IMPORT
import json
import cv2
import subprocess
import base64
import torchaudio
import sys
import shutil
from typing import Generator, Dict, Any, Union # Added Union
import uuid

# Local imports from the same 'services' package or project root
from .audio_processor import AudioProcessor
# ModelManager is NOT directly used by VideoProcessor for translation anymore.
# from .model_manager import ModelManager 
from .error_handler import ErrorHandler
from .translation_strategy import TranslationBackend # Import the ABC

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        logger.info("Initializing VideoProcessor (now relies on passed-in backend for translation)")
        self.audio_processor = AudioProcessor()
        # self.model_manager = ModelManager() # No longer needed for translation here
        # self.processor, self.model, self.text_model, self.tokenizer = self.model_manager.get_model_components() # No longer needed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Can still be used for other torch ops
        
        self.temp_dir = Path('temp_video_processing') # Changed temp dir name for clarity
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"VideoProcessor temp directory: {self.temp_dir.resolve()}")
        
        self._check_wav2lip_setup()

    def _check_wav2lip_setup(self):
        # ... (same as before) ...
        logger.info("Checking Wav2Lip setup...")
        wav2lip_dir = Path("Wav2Lip") # Assuming Wav2Lip is in the Backend directory
        if not wav2lip_dir.is_dir():
            logger.error(f"Wav2Lip directory not found at {wav2lip_dir.resolve()}! Lip sync will fail.")
            return
        # ... (rest of file checks as before)
        files_to_check = [
            (wav2lip_dir / "inference.py", "Inference script"),
            (wav2lip_dir / "checkpoints/wav2lip.pth", "Primary model checkpoint"),
        ]
        all_found = True
        for file_path, description in files_to_check:
            if file_path.exists():
                logger.info(f"✓ {description} found: {file_path}")
            else:
                logger.warning(f"✗ {description} NOT FOUND: {file_path}. Lip sync might fail.")
                all_found = False
        if not all_found:
             logger.error("One or more critical Wav2Lip files are missing. Lip sync is likely to fail.")


    def _progress_event(self, progress: int, phase: str) -> str:
        logger.debug(f"Progress update: {progress}% - {phase}")
        return f"data: {json.dumps({'progress': progress, 'phase': phase})}\n\n"

    def _cleanup_temp_files(self, files_to_clean: list):
        logger.debug(f"Cleaning up {len(files_to_clean)} temporary files/dirs for video processing.")
        for item_path_str in files_to_clean:
            item_path = Path(item_path_str)
            try:
                if item_path.exists():
                    if item_path.is_dir():
                        logger.debug(f"Removing temporary directory: {item_path}")
                        shutil.rmtree(item_path)
                    else:
                        logger.debug(f"Removing temporary file: {item_path}")
                        os.unlink(item_path)
            except Exception as e:
                logger.error(f"Failed to cleanup {item_path}: {str(e)}")
        logger.debug("Temporary file/dir cleanup complete for video processing.")


    # extract_audio, save_audio_tensor, combine_audio_video, detect_speech_start, detect_face, apply_lip_sync
    # These methods remain largely the same as your last correct version of video_routes.py
    # Ensure they use self.temp_dir correctly for creating intermediate files.
    # For brevity, I'll assume they are correct and focus on process_video.

    def extract_audio(self, video_path: str, unique_id: str) -> str:
        logger.info(f"Extracting audio from {video_path}")
        audio_path = str(self.temp_dir / f'{unique_id}_temp_audio.wav')
        try:
            command = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
            subprocess.run(command, capture_output=True, text=True, check=True)
            if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
                raise ValueError("FFmpeg audio extraction failed or produced empty file.")
            logger.info(f"Audio extracted to {audio_path} (size: {Path(audio_path).stat().st_size} bytes)")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed to extract audio: {e.stderr}")
            raise ValueError(f"Failed to extract audio: {e.stderr}")

    def save_audio_tensor(self, audio_tensor: torch.Tensor, path: str, sample_rate: int = 16000):
        logger.info(f"Saving audio tensor of shape {audio_tensor.shape} to {path}")
        try:
            waveform_to_save = audio_tensor.squeeze().cpu() # Ensure it's 1D for torchaudio save
            if waveform_to_save.ndim == 0: # If it became a scalar after squeeze
                waveform_to_save = waveform_to_save.unsqueeze(0) # Make it 1D again
            if waveform_to_save.ndim == 1: # If 1D, add channel dim for torchaudio
                waveform_to_save = waveform_to_save.unsqueeze(0)

            torchaudio.save(path, waveform_to_save, sample_rate=sample_rate)
            if not Path(path).exists() or Path(path).stat().st_size == 0:
                 raise ValueError("Torchaudio save failed or produced empty file.")
            logger.info(f"Audio tensor saved to {path} (size: {Path(path).stat().st_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to save audio tensor to {path}: {str(e)}", exc_info=True)
            raise

    def apply_lip_sync(self, original_video_path: str, translated_audio_path: str, lip_synced_video_path: str):
        # ... (Your existing Wav2Lip command execution logic) ...
        # This method needs careful error handling and logging as it's a common failure point.
        logger.info(f"Starting lip sync with Wav2Lip: Video='{original_video_path}', Audio='{translated_audio_path}'")
        wav2lip_dir = Path("Wav2Lip") # Relative to Backend directory
        checkpoint_path = wav2lip_dir / "checkpoints" / "wav2lip.pth" # Prioritize standard model
        
        if not checkpoint_path.exists():
            logger.warning(f"Wav2Lip checkpoint {checkpoint_path} not found. Trying GAN model...")
            checkpoint_path = wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
            if not checkpoint_path.exists():
                logger.error(f"CRITICAL: No Wav2Lip checkpoint found (neither standard nor GAN). Lip sync will fail.")
                return False # Indicate failure

        command = [
            sys.executable, str(wav2lip_dir / "inference.py"), # Use sys.executable for python
            "--checkpoint_path", str(checkpoint_path),
            "--face", str(original_video_path),
            "--audio", str(translated_audio_path),
            "--outfile", str(lip_synced_video_path),
            "--nosmooth",
            "--pads", "0", "20", "0", "0", # Common padding
            "--resize_factor", "1" # Start with 1, can be tuned
        ]
        logger.info(f"Running Wav2Lip command: {' '.join(command)}")
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False) # check=False to handle errors manually
            if process.returncode != 0:
                logger.error(f"Wav2Lip process failed with code {process.returncode}.")
                logger.error(f"Wav2Lip STDOUT: {process.stdout}")
                logger.error(f"Wav2Lip STDERR: {process.stderr}")
                return False
            if not Path(lip_synced_video_path).exists() or Path(lip_synced_video_path).stat().st_size < 1000: # Basic check
                logger.error(f"Wav2Lip ran but output file is missing or too small: {lip_synced_video_path}")
                logger.error(f"Wav2Lip STDOUT: {process.stdout}")
                logger.error(f"Wav2Lip STDERR: {process.stderr}")
                return False
            logger.info(f"Wav2Lip lip sync successful. Output: {lip_synced_video_path}")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Wav2Lip process timed out after 5 minutes.")
            return False
        except Exception as e:
            logger.error(f"Exception during Wav2Lip execution: {e}", exc_info=True)
            return False

    def combine_audio_video(self, final_audio_path: str, original_video_path: str, output_path: str):
        # Uses the final audio (e.g., from OpenVoice) and merges it with the original video stream
        logger.info(f"Combining audio '{final_audio_path}' with video from '{original_video_path}' into '{output_path}'")
        command = [
            'ffmpeg', '-y',
            '-i', str(original_video_path),    # Input 0 (original video for its video stream)
            '-i', str(final_audio_path),      # Input 1 (final translated & voice-cloned audio)
            '-map', '0:v:0',                  # Select video stream from input 0
            '-map', '1:a:0',                  # Select audio stream from input 1
            '-c:v', 'copy',                   # Copy video codec (no re-encoding)
            '-c:a', 'aac',                    # Encode audio to AAC (common compatible format)
            '-shortest',                      # Finish encoding when the shortest input stream ends
            str(output_path)
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                raise ValueError("FFmpeg final video combination failed or produced empty file.")
            logger.info(f"Final video with translated audio saved to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed to combine final audio and video: {e.stderr}")
            raise ValueError(f"Failed to create final video: {e.stderr}")


    # This is the main generator function called by the route handler
    def process_video(self, video_file, target_language: str, backend_instance: TranslationBackend) -> Generator[str, None, None]:
        request_id = str(uuid.uuid4()) # Unique ID for this processing request
        logger.info(f"[{request_id}] Starting video processing for target_language: {target_language}, using backend: {type(backend_instance).__name__}")
        
        temp_files_created = [] # Keep track of files to clean up
        # Create a unique sub-directory within self.temp_dir for this request's files
        request_temp_dir = self.temp_dir / request_id
        request_temp_dir.mkdir(exist_ok=True)
        temp_files_created.append(str(request_temp_dir)) # Add dir itself for cleanup

        try:
            original_video_filename = secure_filename(video_file.filename)
            temp_video_path = request_temp_dir / original_video_filename
            video_file.save(temp_video_path)
            temp_files_created.append(str(temp_video_path))
            logger.info(f"[{request_id}] Saved uploaded video to: {temp_video_path} (size: {temp_video_path.stat().st_size} bytes)")
            
            yield self._progress_event(10, "Extracting audio from video...")
            original_audio_path = self.extract_audio(str(temp_video_path), request_id) # Pass unique_id for temp naming
            temp_files_created.append(original_audio_path)
        
            yield self._progress_event(20, "Processing original audio...")
            # Process audio for translation (e.g. to tensor)
            audio_tensor = self.audio_processor.process_audio(original_audio_path) # Assuming this returns a tensor
        
            yield self._progress_event(30, f"Translating audio to {target_language}...")
            logger.info(f"[{request_id}] Calling backend.translate_speech for language: {target_language}")
            translation_result = backend_instance.translate_speech(
                audio_tensor=audio_tensor,
                source_lang="eng", # Assuming source is English for video processing
                target_lang=target_language
            )
            translated_audio_tensor = translation_result["audio"]
            source_text = translation_result["transcripts"]["source"]
            target_text = translation_result["transcripts"]["target"]
            logger.info(f"[{request_id}] Translation successful. Source: '{source_text[:50]}...', Target: '{target_text[:50]}...'")

            translated_audio_path = str(request_temp_dir / f"{request_id}_translated_audio.wav")
            self.save_audio_tensor(translated_audio_tensor, translated_audio_path)
            temp_files_created.append(translated_audio_path)
        
            yield self._progress_event(60, "Applying lip synchronization (Wav2Lip)...")
            lip_synced_video_path = request_temp_dir / f"{request_id}_lip_synced_video.mp4"
            
            # Wav2Lip needs the *original* video for visual frames and the *translated* audio
            lip_sync_success = self.apply_lip_sync(str(temp_video_path), translated_audio_path, str(lip_synced_video_path))
            
            final_video_to_serve_path = ""
            if lip_sync_success:
                logger.info(f"[{request_id}] Lip sync successful: {lip_synced_video_path}")
                final_video_to_serve_path = str(lip_synced_video_path)
                temp_files_created.append(str(lip_synced_video_path)) # Add for cleanup
            else:
                logger.warning(f"[{request_id}] Lip sync failed or was skipped. Merging translated audio with original video.")
                # Fallback: Combine translated audio with original video frames without new lip sync
                fallback_video_path = request_temp_dir / f"{request_id}_fallback_video_with_translated_audio.mp4"
                self.combine_audio_video(translated_audio_path, str(temp_video_path), str(fallback_video_path))
                final_video_to_serve_path = str(fallback_video_path)
                temp_files_created.append(str(fallback_video_path)) # Add for cleanup

            if not Path(final_video_to_serve_path).exists() or Path(final_video_to_serve_path).stat().st_size == 0:
                error_msg = f"[{request_id}] Final video file is missing or empty after processing."
                logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg, 'phase': 'Finalization Error'})}\n\n"
                return

            yield self._progress_event(90, "Finalizing and preparing video...")
            with open(final_video_to_serve_path, 'rb') as f_final_video:
                video_data_b64 = base64.b64encode(f_final_video.read()).decode('utf-8')
            
            response_data = {
                'result': video_data_b64,
                'transcripts': {
                    'source': source_text,
                    'target': target_text
                }
            }
            yield f"data: {json.dumps(response_data)}\n\n"
            logger.info(f"[{request_id}] Video processing completed and result sent.")
        
        except Exception as e:
            logger.error(f"[{request_id}] Video processing failed: {str(e)}", exc_info=True)
            error_data = {'error': str(e), 'phase': 'Processing Error'}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            self._cleanup_temp_files(temp_files_created)


# This is the function imported and called by app.py
def handle_video_processing(video_files: Dict[str, Any], target_language: str, backend_instance: TranslationBackend):
    logger.info(f"handle_video_processing called for lang: {target_language} using backend: {type(backend_instance).__name__}")
    
    if 'video' not in video_files:
        logger.error("No 'video' file found in the request files for handle_video_processing.")
        # For SSE, we stream an error event
        def error_stream_no_file():
            error_data = {'error': 'No video file provided in request.', 'phase': 'File Error'}
            yield f"data: {json.dumps(error_data)}\n\n"
        return Response(stream_with_context(error_stream_no_file()), mimetype='text/event-stream', status=400)
        
    video_file_storage = video_files['video'] # This is a FileStorage object
    if not video_file_storage.filename:
        logger.error("Empty video filename in request for handle_video_processing.")
        def error_stream_no_filename():
            error_data = {'error': 'No video file selected (empty filename).', 'phase': 'File Error'}
            yield f"data: {json.dumps(error_data)}\n\n"
        return Response(stream_with_context(error_stream_no_filename()), mimetype='text/event-stream', status=400)
    
    # Instantiate VideoProcessor here, it will use the passed backend_instance
    processor_instance = VideoProcessor() 
    
    return Response(
        stream_with_context(processor_instance.process_video(video_file_storage, target_language, backend_instance)),
        mimetype='text/event-stream'
    )