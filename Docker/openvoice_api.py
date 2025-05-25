from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import os
import time
import tempfile
import torch
import soundfile as sf
import shutil
from typing import Optional, Tuple, Generator 
import logging
import librosa 
import numpy as np 
import torch.nn.functional as F 
import json 
from pathlib import Path

from openvoice.api import ToneColorConverter
from openvoice import se_extractor 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI(title="OpenVoice API", description="API for voice cloning with OpenVoice")

tone_converter_model: Optional[ToneColorConverter] = None
default_source_se: Optional[torch.Tensor] = None 

@app.on_event("startup")
async def startup_event():
   global tone_converter_model, default_source_se
   current_device = "cpu" 
   try:
       logger.info(f"OpenVoice API starting up. Attempting to load models on device: {current_device}")
       config_path_str = "/app/checkpoints_v2/converter/config.json" 
       checkpoint_path_str = "/app/checkpoints_v2/converter/checkpoint.pth"
       config_path = Path(config_path_str); checkpoint_path = Path(checkpoint_path_str)
       if not config_path.exists(): raise FileNotFoundError(f"Config not found: {config_path_str}")
       if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_str}")
       logger.info(f"Reading ToneColorConverter config from: {config_path_str}")
       with open(config_path, 'r') as f: config = json.load(f)
       if "model" in config and "contentvec_final_proj" not in config["model"]:
           logger.warning("! 'contentvec_final_proj' missing in config.json. MANUAL EDIT OF HOST FILE NEEDED & REBUILD DOCKER !")
       else: logger.info("Converter config check: 'contentvec_final_proj' key ok.")
       tone_converter_model = ToneColorConverter(str(config_path), device=current_device)
       tone_converter_model.load_ckpt(str(checkpoint_path))
       logger.info(f"ToneColorConverter model loaded successfully.")
       default_speaker_path_str = "/app/checkpoints_v2/base_speakers/ses/en-us.pth" 
       default_speaker_path = Path(default_speaker_path_str)
       if default_speaker_path.exists():
           logger.info(f"Loading default source SE from: {default_speaker_path_str}")
           default_source_se_raw = torch.load(default_speaker_path, map_location=current_device)
           if default_source_se_raw.shape == (1, 256): default_source_se = F.normalize(default_source_se_raw, p=2, dim=1)
           elif default_source_se_raw.numel() >= 256: 
                default_source_se_flat = default_source_se_raw.flatten()[:256].reshape(1, 256)
                default_source_se = F.normalize(default_source_se_flat, p=2, dim=1)
                logger.info(f"Reshaped default_source_se to {default_source_se.shape}")
           else:
               logger.warning(f"Default SE bad shape: {default_source_se_raw.shape}. Using random."); default_source_se = F.normalize(torch.randn(1, 256, device=current_device), p=2, dim=1)
           logger.info(f"Default source SE loaded, shape: {default_source_se.shape}")
       else:
           logger.warning(f"Default source SE file NOT FOUND: {default_speaker_path_str}. Using random."); default_source_se = F.normalize(torch.randn(1, 256, device=current_device), p=2, dim=1) 
           logger.info(f"Created fallback default_source_se, shape: {default_source_se.shape}")
       logger.info("OpenVoice API: Startup complete.")
   except Exception as e:
       logger.error(f"CRITICAL STARTUP FAILURE OpenVoice API: {str(e)}", exc_info=True)
       tone_converter_model = None; default_source_se = None

@app.get("/")
async def root(): return {"message": "OpenVoice API is running"}

@app.get("/status")
async def status():
    tone_converter_loaded_flag = tone_converter_model is not None
    default_source_se_loaded_flag = default_source_se is not None
    if tone_converter_loaded_flag and default_source_se_loaded_flag:
        return {"tone_converter_model_loaded": True, "default_source_se_loaded": True, "message": "Models loaded."}
    else: 
        logger.error(f"OpenVoice API /status check: Models NOT loaded. Converter: {tone_converter_loaded_flag}, DefaultSE: {default_source_se_loaded_flag}")
        raise HTTPException(status_code=503, detail="Models not loaded.")

def _save_upload_file_to_temp(upload_file: UploadFile, temp_dir_path: Path, desired_filename: str) -> Path: # Takes Path for temp_dir_path
    temp_file_path = temp_dir_path / desired_filename
    try:
        with open(temp_file_path, "wb") as f_buffer: shutil.copyfileobj(upload_file.file, f_buffer)
        logger.info(f"Saved '{upload_file.filename}' to '{temp_file_path}', size: {temp_file_path.stat().st_size}")
        return temp_file_path
    except Exception as e: logger.error(f"Failed to save {upload_file.filename} to {temp_file_path}: {e}", exc_info=True); raise

def _preprocess_audio_for_openvoice(audio_path: Path, temp_dir_path: Path, filename_prefix: str) -> Path: # Takes Path for temp_dir_path
    try:
        logger.info(f"Preprocessing OV audio: {audio_path.name}")
        audio_np, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if len(audio_np) < 1600: raise ValueError(f"Audio '{audio_path.name}' too short (min 0.1s).")
        processed_path = temp_dir_path / f"{filename_prefix}_processed_16k.wav"
        sf.write(str(processed_path), audio_np, 16000)
        logger.info(f"Processed OV audio saved: {processed_path} (Size: {processed_path.stat().st_size})")
        return processed_path
    except Exception as e: logger.error(f"Error processing OV audio {audio_path}: {e}", exc_info=True); raise ValueError(f"Failed to process '{audio_path.name}': {e}")

async def _cleanup_temp_dir_async(temp_dir_path_str: str): # Takes string path for safety with background task
    try:
        if os.path.exists(temp_dir_path_str): 
            shutil.rmtree(temp_dir_path_str)
            logger.info(f"Async cleanup: Successfully cleaned up temp directory: {temp_dir_path_str}")
    except Exception as e:
        logger.error(f"Async cleanup: Error cleaning up temp directory {temp_dir_path_str}: {e}", exc_info=True)

@app.post("/clone-voice")
async def clone_voice(
   reference_audio_file: UploadFile = File(..., description="Audio file of the voice to clone."),
   content_audio_file: UploadFile = File(..., description="Audio file whose content will be spoken in the cloned voice.")
):
   req_id = str(time.time())[-5:] 
   logger.info(f"--- [{req_id}] /clone-voice endpoint CALLED ---")
   logger.info(f"[{req_id}] Reference: {reference_audio_file.filename}, Type: {reference_audio_file.content_type}")
   logger.info(f"[{req_id}] Content: {content_audio_file.filename}, Type: {content_audio_file.content_type}")

   if not tone_converter_model or default_source_se is None:
       logger.error(f"[{req_id}] Models not loaded."); raise HTTPException(status_code=503, detail="Models not available.")
   
   request_temp_dir_str = tempfile.mkdtemp(prefix=f"ov_clone_req_{req_id}_")
   request_temp_dir = Path(request_temp_dir_str) # Convert to Path object for use
   logger.info(f"[{req_id}] Created temporary directory for request: {request_temp_dir}")
   
   output_path: Optional[Path] = None # Define output_path here to ensure it's in scope for FileResponse

   try:
       raw_ref_path = _save_upload_file_to_temp(reference_audio_file, request_temp_dir, "ref_raw.wav")
       proc_ref_path = _preprocess_audio_for_openvoice(raw_ref_path, request_temp_dir, "ref_16k")
       
       raw_content_path = _save_upload_file_to_temp(content_audio_file, request_temp_dir, "content_raw.wav")
       proc_content_path = _preprocess_audio_for_openvoice(raw_content_path, request_temp_dir, "content_16k")
       
       logger.info(f"[{req_id}] Extracting target SE from: {proc_ref_path.name}")
       se_out_dir = request_temp_dir / "se_ref_output"; se_out_dir.mkdir(exist_ok=True) # Use Path object
       
       tgt_se_tensor, _ = se_extractor.get_se(str(proc_ref_path), tone_converter_model, vad=True, target_dir=str(se_out_dir))

       if tgt_se_tensor is None or tgt_se_tensor.numel() == 0: raise ValueError("Reference SE extraction failed.")
       if tgt_se_tensor.shape != (1,256):
           if tgt_se_tensor.numel() >= 256: tgt_se_tensor = tgt_se_tensor.flatten()[:256].reshape(1,256)
           else: raise ValueError(f"Reference SE insufficient elements: {tgt_se_tensor.numel()}")
       tgt_se_norm = F.normalize(tgt_se_tensor.to(default_source_se.device), p=2, dim=1) 
       logger.info(f"[{req_id}] Target SE extracted & normalized, shape: {tgt_se_norm.shape}")

       src_se_3d = default_source_se.unsqueeze(2) if default_source_se.ndim == 2 else default_source_se
       tgt_se_3d = tgt_se_norm.unsqueeze(2) if tgt_se_norm.ndim == 2 else tgt_se_norm
       if src_se_3d.shape!=(1,256,1) or tgt_se_3d.shape!=(1,256,1): raise HTTPException(status_code=500, detail="Internal SE shape error.")

       output_path = request_temp_dir / "cloned_output.wav" # output_path is defined here
       logger.info(f"[{req_id}] Converting '{proc_content_path.name}' to '{output_path.name}' using target SE from '{reference_audio_file.filename}'")
       tone_converter_model.convert(str(proc_content_path), src_se_3d, tgt_se_3d, str(output_path), message="@MyShell")
       
       if not output_path.exists() or output_path.stat().st_size < 1000: 
           logger.error(f"[{req_id}] Cloning produced invalid output. File: {output_path}, Exists: {output_path.exists()}, Size: {output_path.stat().st_size if output_path.exists() else 'N/A'}")
           raise HTTPException(status_code=500, detail="Cloning produced invalid output.")
       
       logger.info(f"[{req_id}] Cloning successful! Output generated at: {output_path}")
       
       # The file needs to be read and sent, then the directory can be cleaned up.
       # Use a generator to stream the file and clean up afterwards.
       async def file_iterator_with_cleanup(file_path_to_stream: Path, dir_to_clean: str) -> Generator[bytes, None, None]:
           try:
               with open(file_path_to_stream, "rb") as f:
                   while chunk := f.read(65536): # Read in 64KB chunks
                       yield chunk
           finally:
               await _cleanup_temp_dir_async(dir_to_clean) # Await the async cleanup

       return StreamingResponse(
           file_iterator_with_cleanup(output_path, request_temp_dir_str), 
           media_type="audio/wav",
           headers={"Content-Disposition": f"attachment; filename=cloned_voice_output.wav"}
       )

   except HTTPException: 
       await _cleanup_temp_dir_async(request_temp_dir_str) # Ensure cleanup on HTTPException before re-raising
       raise 
   except ValueError as ve: 
       logger.warning(f"[{req_id}] Input error: {str(ve)}")
       await _cleanup_temp_dir_async(request_temp_dir_str)
       raise HTTPException(status_code=400, detail=f"Input error: {str(ve)}")
   except Exception as e: 
       logger.error(f"[{req_id}] Critical error in /clone-voice: {str(e)}", exc_info=True)
       await _cleanup_temp_dir_async(request_temp_dir_str) 
       raise HTTPException(status_code=500, detail="Server error during cloning.")