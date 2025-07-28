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
import requests

from .audio_processor import AudioProcessor
from .translation_strategy import TranslationBackend 

MUSETALK_API_URL = os.getenv("MUSETALK_API_URL", "http://musetalk-api:8000")

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

    # REPLACE the old apply_lip_sync method in VideoProcessor class with this one.
    def apply_lip_sync(self, original_video_path: Path, translated_audio_path: Path, lip_synced_video_output_path: Path) -> bool:
        """
        Calls the MuseTalk API to perform lip-sync.
        Returns True on success, False on failure.
        """
        request_id_short = lip_synced_video_output_path.stem.split('_')[0]
        musetalk_url = f"{MUSETALK_API_URL}/lipsync-video/"
        logger.info(f"[{request_id_short}] Calling MuseTalk API for lip sync at: {musetalk_url}")

        try:
            with open(original_video_path, "rb") as video_file, open(translated_audio_path, "rb") as audio_file:
                files = {
                    "video_file": (original_video_path.name, video_file, "video/mp4"),
                    "audio_file": (translated_audio_path.name, audio_file, "audio/wav"),
                }
                data = {"bbox_shift": 0} # This can be parameterized later if needed

                # Use a long timeout as lip-sync can be slow
                response = requests.post(musetalk_url, files=files, data=data, timeout=7200) # 2-hour timeout

            if response.status_code == 200:
                # Save the returned video content directly to the output path
                with open(lip_synced_video_output_path, "wb") as f_out:
                    f_out.write(response.content)
                
                if lip_synced_video_output_path.exists() and lip_synced_video_output_path.stat().st_size > 1000:
                    logger.info(f"[{request_id_short}] MuseTalk API call successful. Saved lip-synced video to {lip_synced_video_output_path.name}")
                    return True
                else:
                    logger.error(f"[{request_id_short}] MuseTalk API returned success but output file is missing or empty.")
                    return False
            else:
                # Log the error response from the MuseTalk API
                error_detail = response.text[:500] # Get first 500 chars of error
                logger.error(f"[{request_id_short}] MuseTalk API call failed with status {response.status_code}. Detail: {error_detail}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"[{request_id_short}] Failed to connect or communicate with MuseTalk API: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[{request_id_short}] An unexpected error occurred during the MuseTalk API call: {e}", exc_info=True)
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
                audio_tensor=audio_tensor, source_lang="eng", target_lang=target_language, original_video_path=temp_original_video_path
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
                current_phase_for_error = "Applying lip synchronization (MuseTalk)"
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