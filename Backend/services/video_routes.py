# services/video_routes.py
import os
import time
import torch
import logging
import tempfile
import numpy as np
from pathlib import Path
from flask import Response, stream_with_context 
from werkzeug.utils import secure_filename
import json
import subprocess
import base64
import torchaudio
import sys 
import shutil
from typing import Generator, Dict, Any, Union, Optional, Tuple
import uuid

from .audio_processor import AudioProcessor
from .translation_strategy import TranslationBackend 

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        logger.info("Initializing VideoProcessor (relies on passed-in backend for translation)")
        self.audio_processor = AudioProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_project_dir = Path(__file__).resolve().parent.parent 
        self.base_temp_dir = self.base_project_dir / 'temp_video_processing_requests' 
        self.base_temp_dir.mkdir(parents=True, exist_ok=True) 
        logger.info(f"VideoProcessor base temp directory for requests: {self.base_temp_dir.resolve()}")
        
        self.wav2lip_path = self.base_project_dir / "Wav2Lip" 
        self._check_wav2lip_setup()

    def _check_wav2lip_setup(self):
        logger.info(f"Checking Wav2Lip setup in: {self.wav2lip_path}")
        if not self.wav2lip_path.is_dir():
            logger.error(f"Wav2Lip directory not found at {self.wav2lip_path}! Lip sync will fail.")
            self.wav2lip_inference_script_abs_path = None 
            self.wav2lip_checkpoint_abs_path = None
            self.wav2lip_gan_checkpoint_abs_path = None
            return

        self.wav2lip_inference_script_abs_path = (self.wav2lip_path / "inference.py").resolve()
        self.wav2lip_checkpoint_abs_path = (self.wav2lip_path / "checkpoints" / "wav2lip.pth").resolve()
        self.wav2lip_gan_checkpoint_abs_path = (self.wav2lip_path / "checkpoints" / "wav2lip_gan.pth").resolve()

        if not self.wav2lip_inference_script_abs_path.exists():
            logger.error(f"Wav2Lip inference script NOT FOUND: {self.wav2lip_inference_script_abs_path}")
        else:
            logger.info(f"Found Wav2Lip inference script: {self.wav2lip_inference_script_abs_path}")

        if not self.wav2lip_checkpoint_abs_path.exists():
            logger.warning(f"Primary Wav2Lip checkpoint NOT FOUND: {self.wav2lip_checkpoint_abs_path}")
            if self.wav2lip_gan_checkpoint_abs_path.exists():
                logger.info(f"Found Wav2Lip GAN checkpoint (will use as fallback): {self.wav2lip_gan_checkpoint_abs_path}")
            else:
                logger.error(f"CRITICAL: NEITHER Wav2Lip standard nor GAN checkpoint found. Lip sync will fail.")
        else:
            logger.info(f"Found Wav2Lip standard checkpoint: {self.wav2lip_checkpoint_abs_path}")

    def _progress_event(self, progress: int, phase: str) -> str:
        logger.debug(f"Progress: {progress}% - {phase}")
        return f"data: {json.dumps({'progress': progress, 'phase': phase})}\n\n"

    def _cleanup_request_files(self, request_dir: Path):
        if request_dir and request_dir.exists() and request_dir.is_dir():
            if not str(request_dir.resolve()).startswith(str(self.base_temp_dir.resolve())):
                logger.error(f"CRITICAL SECURITY: Attempt to delete directory outside base temp path: {request_dir}. Aborting cleanup for this path.")
                return
            logger.info(f"Cleaning up temporary request directory: {request_dir}")
            try:
                shutil.rmtree(request_dir)
                logger.info(f"Successfully removed {request_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup request directory {request_dir}: {str(e)}")
        else:
            logger.debug(f"Cleanup skipped: Temporary request directory {request_dir} not found or not a directory.")

    def extract_audio(self, video_path: Path, request_temp_dir: Path) -> Path:
        audio_output_path = request_temp_dir / f"{video_path.stem}_extracted_audio.wav"
        logger.info(f"Extracting audio from {video_path} to {audio_output_path}")
        try:
            command = ['ffmpeg', '-y', '-i', str(video_path), 
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
                       str(audio_output_path)]
            logger.debug(f"FFmpeg audio extraction command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction failed. STDERR: {result.stderr}")
                raise ValueError(f"FFmpeg audio extraction error: {result.stderr}")
            if not audio_output_path.exists() or audio_output_path.stat().st_size == 0:
                raise ValueError("FFmpeg audio extraction produced empty/no file.")
            logger.info(f"Audio extracted: {audio_output_path} (size: {audio_output_path.stat().st_size} bytes)")
            return audio_output_path
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg audio extraction timed out for {video_path}.")
            raise TimeoutError("Audio extraction timed out.")
        except Exception as e: 
            logger.error(f"General error in extract_audio: {e}", exc_info=True)
            raise

    def save_audio_tensor(self, audio_tensor: torch.Tensor, output_path: Path, sample_rate: int = 16000):
        logger.info(f"Saving audio tensor shape {audio_tensor.shape} to {output_path}")
        try:
            waveform_to_save = audio_tensor.squeeze().cpu()
            if waveform_to_save.ndim == 0: waveform_to_save = waveform_to_save.unsqueeze(0)
            if waveform_to_save.ndim == 1: waveform_to_save = waveform_to_save.unsqueeze(0)
            torchaudio.save(str(output_path), waveform_to_save, sample_rate=sample_rate)
            if not output_path.exists() or output_path.stat().st_size == 0:
                 raise ValueError(f"Torchaudio save failed for {output_path}")
            logger.info(f"Audio tensor saved: {output_path} (size: {output_path.stat().st_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to save audio tensor to {output_path}: {e}", exc_info=True); raise

    def apply_lip_sync(self, original_video_path: Path, translated_audio_path: Path, lip_synced_video_output_path: Path) -> bool:
        request_id_short = lip_synced_video_output_path.stem.split('_')[0] 
        logger.info(f"[{request_id_short}] Wav2Lip Attempt: Video='{original_video_path.name}', Audio='{translated_audio_path.name}' -> Output='{lip_synced_video_output_path.name}'")

        wav2lip_python_path = self.wav2lip_path / "venv_wav2lip" / "bin" / "python"
        if sys.platform == "win32":
            wav2lip_python_path = self.wav2lip_path / "venv_wav2lip" / "Scripts" / "python.exe"
        
        if not wav2lip_python_path.exists():
            logger.error(f"[{request_id_short}] Wav2Lip Python interpreter not found at {wav2lip_python_path}. "
                         "Ensure 'venv_wav2lip' is created correctly in the Wav2Lip directory and its Python path is correct.")
            return False

        if not self.wav2lip_inference_script_abs_path or not self.wav2lip_inference_script_abs_path.exists():
            logger.error(f"[{request_id_short}] Wav2Lip inference.py not found (expected at {self.wav2lip_inference_script_abs_path}). Cannot perform lip sync.")
            return False
        
        checkpoint_to_use_abs_path: Optional[Path] = None
        if self.wav2lip_checkpoint_abs_path and self.wav2lip_checkpoint_abs_path.exists():
            checkpoint_to_use_abs_path = self.wav2lip_checkpoint_abs_path
        elif self.wav2lip_gan_checkpoint_abs_path and self.wav2lip_gan_checkpoint_abs_path.exists():
            checkpoint_to_use_abs_path = self.wav2lip_gan_checkpoint_abs_path
            logger.info(f"[{request_id_short}] Using Wav2Lip GAN checkpoint as standard one was not found.")
        else:
            logger.error(f"[{request_id_short}] CRITICAL: No Wav2Lip checkpoint (.pth file) found. Lip sync cannot proceed.")
            return False
        
        logger.info(f"[{request_id_short}] Using Wav2Lip checkpoint: {checkpoint_to_use_abs_path}")

        abs_original_video_path = str(original_video_path.resolve())
        abs_translated_audio_path = str(translated_audio_path.resolve())
        abs_lip_synced_video_output_path = str(lip_synced_video_output_path.resolve())

        command = [
            str(wav2lip_python_path), 
            str(self.wav2lip_inference_script_abs_path),
            "--checkpoint_path", str(checkpoint_to_use_abs_path),
            "--face", abs_original_video_path,
            "--audio", abs_translated_audio_path,
            "--outfile", abs_lip_synced_video_output_path,
            "--nosmooth", 
            "--resize_factor", "1" # Keep at 1 for now, or make it a parameter
        ]
        logger.info(f"[{request_id_short}] Executing Wav2Lip command: {' '.join(command)}")
        
        wav2lip_working_dir = str(self.wav2lip_path)

        subprocess_env = os.environ.copy()
        keys_to_remove_from_env = ['CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER', 'VIRTUAL_ENV', 'PYTHONHOME']
        for key in keys_to_remove_from_env:
            if key in subprocess_env:
                logger.debug(f"[{request_id_short}] Removing '{key}' from subprocess env. Value was: {subprocess_env[key]}")
                del subprocess_env[key]
        
        venv_bin_path = str((self.wav2lip_path / "venv_wav2lip" / "bin").resolve())
        existing_path = subprocess_env.get('PATH', '')
        subprocess_env['PATH'] = f"{venv_bin_path}{os.pathsep}{existing_path}"
        logger.debug(f"[{request_id_short}] Subprocess PATH for Wav2Lip: {subprocess_env['PATH']}")
        # logger.debug(f"[{request_id_short}] Full subprocess_env keys: {list(subprocess_env.keys())}") # Can be too verbose

        logger.info(f"[{request_id_short}] Starting Wav2Lip subprocess with 2-hour timeout...")
        process_start_time = time.time()
        try:
            process = subprocess.run(command, cwd=wav2lip_working_dir, 
                                     capture_output=True, text=True, 
                                     timeout=7200, # 2 HOURS TIMEOUT
                                     check=False,
                                     env=subprocess_env) 
            
            process_duration = time.time() - process_start_time
            logger.info(f"[{request_id_short}] Wav2Lip subprocess finished in {process_duration:.2f} seconds.")
            
            # Log STDOUT/STDERR regardless of exit code for better debugging
            if process.stdout:
                logger.info(f"[{request_id_short}] Wav2Lip STDOUT:\n{process.stdout}")
            if process.stderr:
                logger.warning(f"[{request_id_short}] Wav2Lip STDERR:\n{process.stderr}")

            if process.returncode != 0:
                logger.error(f"[{request_id_short}] Wav2Lip process failed with code {process.returncode}.")
                return False # Detailed logs already printed
            
            if not lip_synced_video_output_path.exists() or lip_synced_video_output_path.stat().st_size < 10000: 
                size_info = lip_synced_video_output_path.stat().st_size if lip_synced_video_output_path.exists() else 'DOES NOT EXIST'
                logger.error(f"[{request_id_short}] Wav2Lip ran (exit code 0) but output file is missing or too small: {lip_synced_video_output_path} (Size: {size_info})")
                return False
                
            logger.info(f"[{request_id_short}] Wav2Lip lip sync successful. Output: {lip_synced_video_output_path} (Size: {lip_synced_video_output_path.stat().st_size})")
            return True
            
        except subprocess.TimeoutExpired:
            process_duration = time.time() - process_start_time
            logger.error(f"[{request_id_short}] Wav2Lip process timed out after {process_duration:.2f} seconds (limit was 7200s).")
            return False
        except Exception as e:
            process_duration = time.time() - process_start_time
            logger.error(f"[{request_id_short}] Exception during Wav2Lip execution after {process_duration:.2f} seconds: {e}", exc_info=True)
            return False

    def combine_audio_video(self, final_audio_path: Path, original_video_path: Path, output_video_path: Path):
        request_id_short = output_video_path.stem.split('_')[0] 
        logger.info(f"[{request_id_short}] Combining audio '{final_audio_path.name}' with video from '{original_video_path.name}' into '{output_video_path.name}'")
        command = [
            'ffmpeg', '-y',
            '-i', str(original_video_path),   
            '-i', str(final_audio_path),     
            '-map', '0:v:0',                 
            '-map', '1:a:0',                 
            '-c:v', 'copy',                  
            '-c:a', 'aac',                   
            '-shortest',                     
            str(output_video_path)
        ]
        logger.debug(f"[{request_id_short}] FFmpeg combine command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            if result.returncode != 0:
                logger.error(f"[{request_id_short}] FFmpeg combine failed. STDERR: {result.stderr}")
                raise ValueError(f"FFmpeg combine error: {result.stderr}")
            if not output_video_path.exists() or output_video_path.stat().st_size == 0:
                raise ValueError("FFmpeg combine produced empty/no file.")
            logger.info(f"[{request_id_short}] Final video saved by combine_audio_video: {output_video_path}")
        except subprocess.TimeoutExpired:
            logger.error(f"[{request_id_short}] FFmpeg combine_audio_video timed out for {original_video_path.name}.")
            raise TimeoutError("Video and audio combination timed out.")
        except Exception as e: 
            logger.error(f"[{request_id_short}] General error in combine_audio_video: {e}", exc_info=True); raise

    def process_video(self, video_file_storage, target_language: str, backend_instance: TranslationBackend, apply_lip_sync_enabled: bool) -> Generator[str, None, None]:
        request_id = str(uuid.uuid4())
        request_temp_dir = self.base_temp_dir / request_id
        request_temp_dir.mkdir(exist_ok=True)
        phase_logger_str = f"[{request_id}]" 
        logger.info(f"{phase_logger_str} VideoProcessing START. Target: {target_language}, Backend: {type(backend_instance).__name__}, LipSyncEnabled: {apply_lip_sync_enabled}, TempDir: {request_temp_dir}")
        
        temp_paths_to_cleanup = [request_temp_dir] 
        overall_start_time = time.time()
        current_phase_for_error = "Initialization"

        try:
            current_phase_for_error = "Saving uploaded video"
            upload_start_time = time.time()
            original_video_filename = secure_filename(video_file_storage.filename)
            temp_original_video_path = request_temp_dir / original_video_filename 
            video_file_storage.save(str(temp_original_video_path)) 
            temp_paths_to_cleanup.append(temp_original_video_path)
            logger.info(f"{phase_logger_str} Saved uploaded video: {temp_original_video_path.name} ({(time.time()-upload_start_time):.2f}s)")
            
            current_phase_for_error = "Extracting audio"
            yield self._progress_event(10, current_phase_for_error)
            audio_extract_start_time = time.time()
            extracted_original_audio_path = self.extract_audio(temp_original_video_path, request_temp_dir) 
            temp_paths_to_cleanup.append(extracted_original_audio_path)
            logger.info(f"{phase_logger_str} Audio extracted ({(time.time()-audio_extract_start_time):.2f}s)")
        
            current_phase_for_error = "Processing original audio for ASR"
            yield self._progress_event(20, current_phase_for_error)
            asr_process_start_time = time.time()
            audio_tensor = self.audio_processor.process_audio(str(extracted_original_audio_path)) 
            logger.info(f"{phase_logger_str} Original audio processed for ASR ({(time.time()-asr_process_start_time):.2f}s)")
        
            current_phase_for_error = f"Translating to {target_language} & preparing voice"
            yield self._progress_event(30, current_phase_for_error)
            translation_start_time = time.time()
            translation_result = backend_instance.translate_speech(
                audio_tensor=audio_tensor, source_lang="eng", target_lang=target_language
            )
            final_translated_cloned_audio_tensor = translation_result["audio"]
            source_text = translation_result["transcripts"]["source"]
            target_text = translation_result["transcripts"]["target"]
            logger.info(f"{phase_logger_str} Backend translation/voice prep OK ({(time.time()-translation_start_time):.2f}s). Src: '{source_text[:50]}...', Tgt: '{target_text[:50]}...'")

            path_for_final_audio = request_temp_dir / f"{request_id}_final_translated_audio.wav" 
            self.save_audio_tensor(final_translated_cloned_audio_tensor, path_for_final_audio)
            temp_paths_to_cleanup.append(path_for_final_audio)
            logger.info(f"{phase_logger_str} Translated audio saved: {path_for_final_audio.name}")
            
            final_video_to_serve_path: Optional[Path] = None 
            
            if apply_lip_sync_enabled:
                current_phase_for_error = "Applying lip synchronization (Wav2Lip)"
                yield self._progress_event(60, "Applying lip synchronization (this may take some time)...")
                lipsync_start_time = time.time()
                path_for_lip_synced_video = request_temp_dir / f"{request_id}_lip_synced_video.mp4" 
                
                lip_sync_success = self.apply_lip_sync(temp_original_video_path, path_for_final_audio, path_for_lip_synced_video)
                logger.info(f"{phase_logger_str} Lip sync attempt finished in ({(time.time()-lipsync_start_time):.2f}s). Success: {lip_sync_success}")
                
                if lip_sync_success and path_for_lip_synced_video.exists() and path_for_lip_synced_video.stat().st_size > 10000:
                    logger.info(f"{phase_logger_str} Lip sync successful: {path_for_lip_synced_video.name}")
                    final_video_to_serve_path = path_for_lip_synced_video
                else:
                    logger.warning(f"{phase_logger_str} Lip sync failed or output invalid. Fallback: Merging translated audio with original video.")
                    current_phase_for_error = "Lip sync fallback: Combining audio"
                    yield self._progress_event(75, "Lip sync failed, preparing fallback video...") 
                    fallback_start_time = time.time()
                    path_for_fallback_video = request_temp_dir / f"{request_id}_audio_dubbed_no_lipsync.mp4"
                    self.combine_audio_video(path_for_final_audio, temp_original_video_path, path_for_fallback_video)
                    final_video_to_serve_path = path_for_fallback_video
                    logger.info(f"{phase_logger_str} Fallback video created ({(time.time()-fallback_start_time):.2f}s)")
            else:
                logger.info(f"{phase_logger_str} Lip sync was disabled by user. Combining translated audio with original video.")
                current_phase_for_error = "Combining audio (Lip Sync Disabled)"
                yield self._progress_event(75, "Lip sync disabled, combining audio with video...")
                combine_start_time = time.time()
                path_for_combined_video = request_temp_dir / f"{request_id}_audio_dubbed_no_lipsync.mp4"
                self.combine_audio_video(path_for_final_audio, temp_original_video_path, path_for_combined_video)
                final_video_to_serve_path = path_for_combined_video
                logger.info(f"{phase_logger_str} Video with new audio (no lip sync) created ({(time.time()-combine_start_time):.2f}s)")


            if not final_video_to_serve_path or not final_video_to_serve_path.exists() or final_video_to_serve_path.stat().st_size < 10000: 
                error_msg = f"{phase_logger_str} Final video output missing/empty after processing."
                logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg,'phase': 'Finalization Error'})}\n\n"; return

            current_phase_for_error = "Finalizing video (encoding)"
            yield self._progress_event(90, current_phase_for_error)
            finalization_start_time = time.time()
            with open(final_video_to_serve_path, 'rb') as f_final_video:
                video_data_b64 = base64.b64encode(f_final_video.read()).decode('utf-8')
            logger.info(f"{phase_logger_str} Video finalized and encoded ({(time.time()-finalization_start_time):.2f}s)")
            
            yield f"data: {json.dumps({'result': video_data_b64, 'transcripts': {'source': source_text, 'target': target_text}})}\n\n"
            logger.info(f"{phase_logger_str} Video processing COMPLETED & result sent (Total time: {(time.time()-overall_start_time):.2f}s)")
        
        except Exception as e:
            phase_at_error = locals().get('current_phase_for_error', "Unknown Error Phase during processing")
            logger.error(f"{phase_logger_str} Video processing FAILED during '{phase_at_error}': {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'phase': f'Error during: {phase_at_error}'})}\n\n"
        finally:
            logger.info(f"{phase_logger_str} Entering finally block for cleanup. Total processing time before cleanup: {(time.time()-overall_start_time):.2f}s")
            self._cleanup_request_files(request_temp_dir) 

# MODIFIED: Added apply_lip_sync to the function signature
def handle_video_processing(request_files: Dict[str, Any], target_language: str, backend_instance: TranslationBackend, apply_lip_sync: bool):
    logger.info(f"handle_video_processing for lang: {target_language}, using backend: {type(backend_instance).__name__}, LipSync: {apply_lip_sync}")
    
    if 'video' not in request_files:
        logger.error("No 'video' in request_files."); 
        def error_stream_no_file(): yield f"data: {json.dumps({'error': 'No video file provided.', 'phase': 'File Upload Error'})}\n\n"
        return Response(stream_with_context(error_stream_no_file()),mimetype='text/event-stream', status=400)
    
    video_file_storage = request_files['video'] 
    if not video_file_storage.filename:
        logger.error("Empty video filename."); 
        def error_stream_no_filename(): yield f"data: {json.dumps({'error': 'No file selected (empty filename).', 'phase': 'File Upload Error'})}\n\n"
        return Response(stream_with_context(error_stream_no_filename()),mimetype='text/event-stream', status=400)
    
    processor_instance = VideoProcessor() 
    # MODIFIED: Pass apply_lip_sync to the processor's method
    return Response(
        stream_with_context(processor_instance.process_video(video_file_storage, target_language, backend_instance, apply_lip_sync)), 
        mimetype='text/event-stream'
    )