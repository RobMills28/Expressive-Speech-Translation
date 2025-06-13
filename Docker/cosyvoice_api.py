# Docker/cosyvoice_api.py
import sys
import os
import torchaudio
from pathlib import Path
import tempfile
import shutil
import logging
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Generator

# Ensure CosyVoice modules can be found
# Assumes this script is in /app/ and CosyVoice is in /app/CosyVoice/
sys.path.append('/app/CosyVoice/third_party/Matcha-TTS') # From CosyVoice examples

# --- MODIFIED: Import parent CosyVoice class ---
from cosyvoice.cli.cosyvoice import CosyVoice # Was CosyVoice2
from cosyvoice.utils.file_utils import load_wav # Used by CosyVoice for loading prompt

# --- Logging Setup ---
# Ensure logging is configured before any loggers are potentially used by imported modules at global scope
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d (%(funcName)s)] - %(message)s')
logger = logging.getLogger(__name__) # Logger for this API script

app = FastAPI(title="CosyVoice API", description="API for Text-to-Speech with voice cloning using CosyVoice.")

# --- Global Model Variable ---
# --- MODIFIED: Type hint and MODEL_PATH for CosyVoice-300M ---
cosy_model: Optional[CosyVoice] = None
MODEL_PATH = "/app/CosyVoice/pretrained_models/CosyVoice-300M" # Path to CosyVoice-300M

@app.on_event("startup")
async def startup_event():
    global cosy_model
    logger.info(f"CosyVoice API FastAPI startup_event: Attempting to load MODEL: {MODEL_PATH}")
    model_dir_path = Path(MODEL_PATH)

    if not model_dir_path.exists() or not model_dir_path.is_dir():
        logger.error(f"CRITICAL (startup_event): CosyVoice model directory not found or not a directory: {MODEL_PATH}")
        cosy_model = None # Ensure model is None
        return # Exit startup if path is bad
    
    expected_yaml = model_dir_path / "cosyvoice.yaml"
    expected_llm_pt = model_dir_path / "llm.pt"

    if not expected_yaml.exists():
        logger.error(f"CRITICAL (startup_event): Expected config file {expected_yaml.name} not found in {MODEL_PATH}")
        cosy_model = None
        return
    if not expected_llm_pt.exists():
        logger.error(f"CRITICAL (startup_event): Expected model weights file {expected_llm_pt.name} not found in {MODEL_PATH}")
        cosy_model = None
        return

    logger.info(f"CosyVoice API (startup_event): Model directory {MODEL_PATH} and key files ({expected_yaml.name}, {expected_llm_pt.name}) exist. Proceeding with model instantiation...")

    try:
        # Parameters for CosyVoice parent class are simpler: model_dir, load_jit, load_trt, fp16
        cosy_model_instance = CosyVoice(MODEL_PATH, load_jit=False, load_trt=False, fp16=False)
        
        # Check if instantiation actually returned a valid model object
        if cosy_model_instance is not None and hasattr(cosy_model_instance, 'sample_rate'):
            cosy_model = cosy_model_instance # Assign to global if successful
            model_name_for_log = Path(MODEL_PATH).name
            logger.info(f"SUCCESS (startup_event): CosyVoice model ({model_name_for_log}) loaded successfully. Model sample rate: {cosy_model.sample_rate}")
        else:
            model_name_for_log = Path(MODEL_PATH).name
            logger.error(f"FAILURE (startup_event): CosyVoice instantiation for model ({model_name_for_log}) returned None or invalid object.")
            cosy_model = None
            
    except Exception as e:
        model_name_for_log = Path(MODEL_PATH).name
        logger.error(f"CRITICAL ERROR (startup_event) loading CosyVoice model ({model_name_for_log}): {e}", exc_info=True)
        cosy_model = None # Ensure model is None if loading failed

@app.get("/health")
async def health_check():
    model_name_for_log = Path(MODEL_PATH).name
    if cosy_model is not None and hasattr(cosy_model, 'sample_rate'):
        return JSONResponse(content={"status": "healthy", "message": f"CosyVoice model ({model_name_for_log}) loaded."})
    else:
        # Provide more detail if model load was attempted but failed from startup_event
        startup_log_check = "CosyVoice model object is None after startup attempt."
        if cosy_model is None and 'cosy_model_instance' in locals() and cosy_model_instance is None: # Check if startup assigned None
             startup_log_check = "CosyVoice instantiation in startup returned None or invalid object."
        
        logger.warning(f"/health check: Model ({model_name_for_log}) is not ready. Current state: {startup_log_check}")
        return JSONResponse(content={"status": "unhealthy", "message": f"CosyVoice model ({model_name_for_log}) not loaded or failed to load. Detail: {startup_log_check}"}, status_code=503)

# ... (the rest of your _save_upload_file_to_temp, _cleanup_temp_dir_async, and generate_speech endpoint functions remain the same as you provided) ...
# ... I will include them here for completeness of THIS file ...

def _save_upload_file_to_temp(upload_file: UploadFile, temp_dir_path: Path) -> Path:
    try:
        suffix = Path(upload_file.filename if upload_file.filename else "audio").suffix or ".wav"
        temp_file_path = temp_dir_path / f"{uuid.uuid4()}{suffix}"
        with open(temp_file_path, "wb") as f_buffer:
            shutil.copyfileobj(upload_file.file, f_buffer)
        logger.info(f"Saved uploaded file '{upload_file.filename}' to '{temp_file_path}' (size: {temp_file_path.stat().st_size})")
        return temp_file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file {upload_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving uploaded file.")
    finally:
        if hasattr(upload_file, 'file') and upload_file.file: 
            upload_file.file.close()

async def _cleanup_temp_dir_async(temp_dir_path_str: str):
    try:
        if os.path.exists(temp_dir_path_str):
            shutil.rmtree(temp_dir_path_str)
            logger.info(f"Async cleanup: Successfully cleaned up temp directory: {temp_dir_path_str}")
    except Exception as e:
        logger.error(f"Async cleanup: Error cleaning up temp directory {temp_dir_path_str}: {e}", exc_info=True)

@app.post("/generate-speech/")
async def generate_speech(
    text_to_synthesize: str = Form(...),
    target_language_code: str = Form(...), 
    reference_speaker_wav: UploadFile = File(...),
    style_prompt_text: Optional[str] = Form("") 
):
    req_id = str(uuid.uuid4())[:8]
    model_name_for_log = Path(MODEL_PATH).name
    logger.info(f"--- [{req_id}] /generate-speech CALLED (Model: {model_name_for_log}) ---")
    logger.info(f"[{req_id}] Target Lang Code (for API): {target_language_code}, Ref Speaker: {reference_speaker_wav.filename}, Text: '{text_to_synthesize[:50]}...'")

    if cosy_model is None:
        logger.error(f"[{req_id}] CosyVoice model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="TTS Service not available: Model not loaded.")

    temp_dir_path_str = tempfile.mkdtemp(prefix=f"cosy_api_req_{req_id}_")
    temp_dir = Path(temp_dir_path_str)
    logger.info(f"[{req_id}] Created temporary directory: {temp_dir}")

    output_wav_path: Optional[Path] = None

    try:
        speaker_wav_path = _save_upload_file_to_temp(reference_speaker_wav, temp_dir)
        
        try:
            prompt_speech_16k_np = load_wav(str(speaker_wav_path), 16000) 
            logger.info(f"[{req_id}] Loaded reference speaker WAV for prompt_speech_16k, shape: {prompt_speech_16k_np.shape if prompt_speech_16k_np is not None else 'None'}")
            if prompt_speech_16k_np is None:
                raise ValueError("load_wav returned None for reference speaker audio.")
        except Exception as e_load_wav:
            logger.error(f"[{req_id}] Failed to load reference speaker WAV {speaker_wav_path} with cosyvoice.utils.load_wav: {e_load_wav}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid reference speaker audio file: {e_load_wav}")

        logger.info(f"[{req_id}] Calling CosyVoice.inference_zero_shot. Output SR will be {cosy_model.sample_rate}Hz.")
        
        generated_chunks = []
        for i, result_chunk in enumerate(cosy_model.inference_zero_shot(
            text=text_to_synthesize, 
            prompt_text=style_prompt_text if style_prompt_text else "", 
            prompt_speech_16k=prompt_speech_16k_np, 
            stream=False 
        )):
            if 'tts_speech' not in result_chunk or result_chunk['tts_speech'] is None:
                logger.warning(f"[{req_id}] Chunk {i} from CosyVoice inference missing 'tts_speech' or it's None. Skipping.")
                continue
            generated_chunks.append(result_chunk['tts_speech']) 
            logger.info(f"[{req_id}] Received chunk {i}, len: {result_chunk['tts_speech'].shape[1]/cosy_model.sample_rate:.2f}s")
        
        if not generated_chunks:
            logger.error(f"[{req_id}] CosyVoice inference returned no valid audio chunks.")
            raise HTTPException(status_code=500, detail="Speech synthesis produced no audio.")

        full_speech_tensor = torch.cat(generated_chunks, dim=1)

        output_wav_path = temp_dir / f"cosyvoice_output_{req_id}.wav"
        save_tensor = full_speech_tensor.cpu()
        if save_tensor.ndim == 1:
            save_tensor = save_tensor.unsqueeze(0)
        
        torchaudio.save(str(output_wav_path), save_tensor, cosy_model.sample_rate)
        
        if not output_wav_path.exists() or output_wav_path.stat().st_size < 100: 
            logger.error(f"[{req_id}] Synthesis produced invalid output file: {output_wav_path}")
            raise HTTPException(status_code=500, detail="Synthesis produced invalid output file.")

        logger.info(f"[{req_id}] Synthesis successful! Output: {output_wav_path} (Size: {output_wav_path.stat().st_size}, SR: {cosy_model.sample_rate}Hz)")
        
        async def file_iterator_with_cleanup(file_path: Path, dir_to_clean: str) -> Generator[bytes, None, None]:
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536): 
                        yield chunk
            finally:
                await _cleanup_temp_dir_async(dir_to_clean)

        return StreamingResponse(
            file_iterator_with_cleanup(output_wav_path, temp_dir_path_str),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=cosyvoice_cloned_{req_id}.wav"}
        )

    except HTTPException: 
        if temp_dir_path_str and os.path.exists(temp_dir_path_str): 
            await _cleanup_temp_dir_async(temp_dir_path_str)
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error during speech generation: {e}", exc_info=True)
        if temp_dir_path_str and os.path.exists(temp_dir_path_str): 
            await _cleanup_temp_dir_async(temp_dir_path_str)
        raise HTTPException(status_code=500, detail=f"Error during speech generation: {str(e)}")