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
import uuid

from openvoice.api import ToneColorConverter
from openvoice import se_extractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI(title="OpenVoice API", description="API for voice cloning with OpenVoice")

tone_converter_model: Optional[ToneColorConverter] = None
default_source_se_for_conversion: Optional[torch.Tensor] = None # Will be [1, 256, 1]

def log_se_properties(se_tensor: Optional[torch.Tensor], name: str, req_id: str = "startup"):
    if se_tensor is None:
        logger.warning(f"[{req_id}] SE '{name}' is None.")
        return
    logger.info(f"[{req_id}] SE '{name}': shape={se_tensor.shape}, dtype={se_tensor.dtype}, device={se_tensor.device}, "
                f"min={se_tensor.min().item():.4f}, max={se_tensor.max().item():.4f}, mean={se_tensor.mean().item():.4f}, std={se_tensor.std().item():.4f}")

@app.on_event("startup")
async def startup_event():
   global tone_converter_model, default_source_se_for_conversion # Renamed
   current_device = "cpu"
   try:
       logger.info(f"OpenVoice API starting up. Attempting to load models on device: {current_device}")
       config_path_str = "/app/checkpoints_v2/converter/config.json"
       checkpoint_path_str = "/app/checkpoints_v2/converter/checkpoint.pth"
       config_path = Path(config_path_str); checkpoint_path = Path(checkpoint_path_str)
       if not config_path.exists(): raise FileNotFoundError(f"Config not found: {config_path_str}")
       if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_str}")
       logger.info(f"Reading ToneColorConverter config from: {config_path_str}")
       with open(config_path, 'r') as f: config_data = json.load(f)
       if "model" in config_data and "contentvec_final_proj" not in config_data["model"]:
           logger.warning("! 'contentvec_final_proj' missing in config.json. MANUAL EDIT OF HOST FILE NEEDED & REBUILD DOCKER !")
       else: logger.info("Converter config check: 'contentvec_final_proj' key ok.")
       
       loaded_gin_channels = config_data.get("model", {}).get("gin_channels", "NOT_FOUND")
       logger.info(f"Config check: gin_channels from loaded config.json is: {loaded_gin_channels}")
       if loaded_gin_channels != 256:
           logger.error(f"CRITICAL CONFIG MISMATCH: gin_channels in config.json is {loaded_gin_channels}, expected 256. This will cause runtime errors.")

       tone_converter_model = ToneColorConverter(str(config_path), device=current_device)
       tone_converter_model.load_ckpt(str(checkpoint_path))
       logger.info(f"ToneColorConverter model loaded successfully.")
       
       if hasattr(tone_converter_model, 'model') and hasattr(tone_converter_model.model, 'hps') and hasattr(tone_converter_model.model.hps, 'data') and hasattr(tone_converter_model.model.hps.data, 'gin_channels'):
           model_hps_gin_channels = tone_converter_model.model.hps.data.gin_channels
           logger.info(f"Model HParams check: gin_channels from loaded model.hps.data.gin_channels is: {model_hps_gin_channels}")
           if model_hps_gin_channels != 256:
                logger.error(f"CRITICAL HPARAM MISMATCH: gin_channels in model HParams is {model_hps_gin_channels}, expected 256.")
       else:
           logger.warning("Could not access model.hps.data.gin_channels for verification.")

       default_speaker_path_str = "/app/checkpoints_v2/base_speakers/ses/en-us.pth"
       default_speaker_path = Path(default_speaker_path_str)
       if default_speaker_path.exists():
           logger.info(f"Loading default source SE from: {default_speaker_path_str}")
           default_source_se_raw = torch.load(default_speaker_path, map_location=current_device)
           log_se_properties(default_source_se_raw, "Default_SE_Raw_at_startup")

           processed_se_2d: Optional[torch.Tensor] = None
           if default_source_se_raw.ndim == 3 and default_source_se_raw.shape[0] == 1 and default_source_se_raw.shape[1] == 256 and default_source_se_raw.shape[2] == 1:
               processed_se_2d = default_source_se_raw.squeeze(-1) # Becomes [1, 256]
           elif default_source_se_raw.ndim == 2 and default_source_se_raw.shape[0] == 1 and default_source_se_raw.shape[1] == 256:
               processed_se_2d = default_source_se_raw # Already [1, 256]
           elif default_source_se_raw.numel() >= 256:
                processed_se_2d = default_source_se_raw.flatten()[:256].reshape(1, 256)
                logger.info(f"Reshaped default_source_se_raw from {default_source_se_raw.shape} to {processed_se_2d.shape}")
           else:
               logger.warning(f"Default SE bad shape: {default_source_se_raw.shape}. Using random.");
               processed_se_2d = torch.randn(1, 256, device=current_device)
           
           # Normalize the [1, 256] tensor
           normalized_2d_se = F.normalize(processed_se_2d, p=2, dim=1)
           # Store as [1, 256, 1] for direct use in convert if its internal logic is tricky
           default_source_se_for_conversion = normalized_2d_se.unsqueeze(-1)
           log_se_properties(default_source_se_for_conversion, "Default_SE_for_Conversion_at_startup") # Shape [1, 256, 1]
       else:
           logger.warning(f"Default source SE file NOT FOUND: {default_speaker_path_str}. Using random.");
           random_2d_se = F.normalize(torch.randn(1, 256, device=current_device), p=2, dim=1)
           default_source_se_for_conversion = random_2d_se.unsqueeze(-1) # Store as [1, 256, 1]
           log_se_properties(default_source_se_for_conversion, "Default_SE_Fallback_Random_at_startup")
       logger.info("OpenVoice API: Startup complete.")
   except Exception as e:
       logger.error(f"CRITICAL STARTUP FAILURE OpenVoice API: {str(e)}", exc_info=True)
       tone_converter_model = None; default_source_se_for_conversion = None

# ... (rest of the file, _save_upload_file_to_temp, _preprocess_audio_for_openvoice, _cleanup_temp_dir_async remains the same) ...

@app.get("/")
async def root(): return {"message": "OpenVoice API is running"}

@app.get("/status")
async def status():
    tone_converter_loaded_flag = tone_converter_model is not None
    default_source_se_loaded_flag = default_source_se_for_conversion is not None
    if tone_converter_loaded_flag and default_source_se_loaded_flag:
        return {"tone_converter_model_loaded": True, "default_source_se_loaded": True, "message": "Models loaded."}
    else:
        logger.error(f"OpenVoice API /status check: Models NOT loaded. Converter: {tone_converter_loaded_flag}, DefaultSE: {default_source_se_loaded_flag}")
        raise HTTPException(status_code=503, detail="Models not loaded.")

def _save_upload_file_to_temp(upload_file: UploadFile, temp_dir_path: Path, desired_filename: str) -> Path:
    temp_file_path = temp_dir_path / desired_filename
    try:
        with open(temp_file_path, "wb") as f_buffer: shutil.copyfileobj(upload_file.file, f_buffer)
        logger.info(f"Saved '{upload_file.filename}' to '{temp_file_path}', size: {temp_file_path.stat().st_size}")
        return temp_file_path
    except Exception as e: logger.error(f"Failed to save {upload_file.filename} to {temp_file_path}: {e}", exc_info=True); raise

def _preprocess_audio_for_openvoice(audio_path: Path, temp_dir_path: Path, filename_prefix: str) -> Path:
    try:
        logger.info(f"Preprocessing OV audio: {audio_path.name}")
        audio_np, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if len(audio_np) < 1600: raise ValueError(f"Audio '{audio_path.name}' too short (min 0.1s @ 16kHz).")
        processed_path = temp_dir_path / f"{filename_prefix}_processed_16k.wav"
        sf.write(str(processed_path), audio_np, 16000)
        logger.info(f"Processed OV audio saved: {processed_path} (Size: {processed_path.stat().st_size})")
        return processed_path
    except Exception as e: logger.error(f"Error processing OV audio {audio_path}: {e}", exc_info=True); raise ValueError(f"Failed to process '{audio_path.name}': {e}")

async def _cleanup_temp_dir_async(temp_dir_path_str: str):
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
   req_id = str(uuid.uuid4())[:8]
   logger.info(f"--- [{req_id}] /clone-voice endpoint CALLED ---")
   logger.info(f"[{req_id}] Reference: {reference_audio_file.filename}, Type: {reference_audio_file.content_type}")
   logger.info(f"[{req_id}] Content: {content_audio_file.filename}, Type: {content_audio_file.content_type}")

   if not tone_converter_model or default_source_se_for_conversion is None: # Check renamed variable
       logger.error(f"[{req_id}] Models not loaded."); raise HTTPException(status_code=503, detail="Models not available.")

   request_temp_dir_str = tempfile.mkdtemp(prefix=f"ov_clone_req_{req_id}_")
   request_temp_dir = Path(request_temp_dir_str)
   logger.info(f"[{req_id}] Created temporary directory for request: {request_temp_dir}")

   output_path: Optional[Path] = None

   try:
       raw_ref_path = _save_upload_file_to_temp(reference_audio_file, request_temp_dir, "ref_raw.wav")
       proc_ref_path = _preprocess_audio_for_openvoice(raw_ref_path, request_temp_dir, "ref_16k")

       raw_content_path = _save_upload_file_to_temp(content_audio_file, request_temp_dir, "content_raw.wav")
       proc_content_path = _preprocess_audio_for_openvoice(raw_content_path, request_temp_dir, "content_16k")

       logger.info(f"[{req_id}] Extracting target SE from: {proc_ref_path.name}")
       se_out_dir = request_temp_dir / "se_ref_output"; se_out_dir.mkdir(exist_ok=True)

       tgt_se_tensor_raw, _ = se_extractor.get_se(
           str(proc_ref_path),
           tone_converter_model,
           vad=True,
           target_dir=str(se_out_dir)
       )
       log_se_properties(tgt_se_tensor_raw, "Target_SE_Raw_from_Extractor", req_id)

       if tgt_se_tensor_raw is None or tgt_se_tensor_raw.numel() == 0:
           raise ValueError("Reference SE extraction failed (returned None or empty).")
       
       tgt_se_tensor_on_device = tgt_se_tensor_raw.to(tone_converter_model.device)

       processed_tgt_se_2d: Optional[torch.Tensor] = None
       if tgt_se_tensor_on_device.ndim == 3 and tgt_se_tensor_on_device.shape[0] == 1 and tgt_se_tensor_on_device.shape[1] == 256 and tgt_se_tensor_on_device.shape[2] == 1:
           processed_tgt_se_2d = tgt_se_tensor_on_device.squeeze(-1)
       elif tgt_se_tensor_on_device.ndim == 2 and tgt_se_tensor_on_device.shape[0] == 1 and tgt_se_tensor_on_device.shape[1] == 256:
           processed_tgt_se_2d = tgt_se_tensor_on_device
       elif tgt_se_tensor_on_device.numel() >= 256:
            processed_tgt_se_2d = tgt_se_tensor_on_device.flatten()[:256].reshape(1, 256)
            logger.info(f"[{req_id}] Reshaped raw target SE from {tgt_se_tensor_on_device.shape} to {processed_tgt_se_2d.shape}")
       else:
           raise ValueError(f"Reference SE insufficient elements after extraction: {tgt_se_tensor_on_device.numel()}")

       tgt_se_norm_2d = F.normalize(processed_tgt_se_2d, p=2, dim=1)
       # Convert to [1, 256, 1] for the convert method, as OpenVoice's own examples often show this shape for SEs
       current_tgt_se_for_conversion = tgt_se_norm_2d.unsqueeze(-1)
       log_se_properties(current_tgt_se_for_conversion, "Target_SE_for_Conversion", req_id)
       
       current_src_se_for_conversion = default_source_se_for_conversion # Already [1, 256, 1]

       output_path = request_temp_dir / "cloned_output.wav"
       logger.info(f"[{req_id}] Converting '{proc_content_path.name}' to '{output_path.name}' using target SE from '{reference_audio_file.filename}'")
       logger.info(f"[{req_id}] src_se shape for convert: {current_src_se_for_conversion.shape}, tgt_se shape for convert: {current_tgt_se_for_conversion.shape}")

       tone_converter_model.convert(
           audio_src_path=str(proc_content_path),
           src_se=current_src_se_for_conversion, # Use explicitly shaped [1, 256, 1]
           tgt_se=current_tgt_se_for_conversion, # Use explicitly shaped [1, 256, 1]
           output_path=str(output_path)
       )

       if not output_path.exists() or output_path.stat().st_size < 1000:
           logger.error(f"[{req_id}] Cloning produced invalid output. File: {output_path}, Exists: {output_path.exists()}, Size: {output_path.stat().st_size if output_path.exists() else 'N/A'}")
           raise HTTPException(status_code=500, detail="Cloning produced invalid output.")

       logger.info(f"[{req_id}] Cloning successful! Output generated at: {output_path} (Size: {output_path.stat().st_size})")

       async def file_iterator_with_cleanup(file_path_to_stream: Path, dir_to_clean: str) -> Generator[bytes, None, None]:
           try:
               with open(file_path_to_stream, "rb") as f:
                   while chunk := f.read(65536):
                       yield chunk
           finally:
               await _cleanup_temp_dir_async(dir_to_clean)

       return StreamingResponse(
           file_iterator_with_cleanup(output_path, request_temp_dir_str),
           media_type="audio/wav",
           headers={"Content-Disposition": f"attachment; filename=cloned_voice_output_{req_id}.wav"}
       )

   except HTTPException:
       await _cleanup_temp_dir_async(request_temp_dir_str)
       raise
   except ValueError as ve:
       logger.warning(f"[{req_id}] Input/Processing error: {str(ve)}", exc_info=True)
       await _cleanup_temp_dir_async(request_temp_dir_str)
       raise HTTPException(status_code=400, detail=f"Input or processing error: {str(ve)}")
   except Exception as e:
       logger.error(f"[{req_id}] Critical error in /clone-voice: {str(e)}", exc_info=True)
       await _cleanup_temp_dir_async(request_temp_dir_str)
       raise HTTPException(status_code=500, detail="Server error during cloning.")