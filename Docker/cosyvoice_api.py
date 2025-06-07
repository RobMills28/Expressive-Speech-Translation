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

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav # Used by CosyVoice for loading prompt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice API", description="API for Text-to-Speech with voice cloning using CosyVoice.")

# --- Global Model Variable ---
cosy_model: Optional[CosyVoice2] = None
MODEL_PATH = "/app/CosyVoice/pretrained_models/CosyVoice2-0.5B" # Path inside Docker

@app.on_event("startup")
async def startup_event():
    global cosy_model
    logger.info("CosyVoice API starting up...")
    if not Path(MODEL_PATH).exists():
        logger.error(f"CRITICAL: CosyVoice model path not found: {MODEL_PATH}")
        # In a real scenario, you might want the app to not start or be unhealthy
        return

    try:
        # fp16 might need GPU, ensure Dockerfile has correct torch for CPU/GPU
        # For CPU deployment, fp16=False is safer.
        # load_jit, load_trt, load_vllm are for optimized inference, start with False.
        cosy_model = CosyVoice2(MODEL_PATH, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        logger.info(f"CosyVoice2-0.5B model loaded successfully. Model sample rate: {cosy_model.sample_rate if cosy_model else 'N/A'}")
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading CosyVoice model: {e}", exc_info=True)
        cosy_model = None # Ensure it's None if loading failed

@app.get("/health")
async def health_check():
    if cosy_model is not None:
        return JSONResponse(content={"status": "healthy", "message": "CosyVoice model loaded."})
    else:
        return JSONResponse(content={"status": "unhealthy", "message": "CosyVoice model not loaded or failed to load."}, status_code=503)

def _save_upload_file_to_temp(upload_file: UploadFile, temp_dir_path: Path) -> Path:
    try:
        # Use a unique name to avoid collisions if multiple files are processed by one request (though not the case here)
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
        if upload_file.file:
            upload_file.file.close()

async def _cleanup_temp_dir_async(temp_dir_path_str: str):
    # ... (same as in previous openvoice_api.py)
    try:
        if os.path.exists(temp_dir_path_str):
            shutil.rmtree(temp_dir_path_str)
            logger.info(f"Async cleanup: Successfully cleaned up temp directory: {temp_dir_path_str}")
    except Exception as e:
        logger.error(f"Async cleanup: Error cleaning up temp directory {temp_dir_path_str}: {e}", exc_info=True)

@app.post("/generate-speech/")
async def generate_speech(
    text_to_synthesize: str = Form(...),
    target_language_code: str = Form(...), # e.g., "es", "fr", "en" (CosyVoice internal codes)
    reference_speaker_wav: UploadFile = File(...),
    style_prompt_text: Optional[str] = Form("") # Optional text prompt for style
):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"--- [{req_id}] /generate-speech CALLED ---")
    logger.info(f"[{req_id}] Target Lang: {target_language_code}, Ref Speaker: {reference_speaker_wav.filename}, Text: '{text_to_synthesize[:50]}...'")

    if cosy_model is None:
        logger.error(f"[{req_id}] CosyVoice model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="TTS Service not available: Model not loaded.")

    temp_dir_path_str = tempfile.mkdtemp(prefix=f"cosy_api_req_{req_id}_")
    temp_dir = Path(temp_dir_path_str)
    logger.info(f"[{req_id}] Created temporary directory: {temp_dir}")

    output_wav_path: Optional[Path] = None

    try:
        # Save reference speaker WAV
        speaker_wav_path = _save_upload_file_to_temp(reference_speaker_wav, temp_dir)
        
        # Load prompt speech at 16kHz as per CosyVoice examples for `prompt_speech_16k`
        try:
            prompt_speech_16k_np = load_wav(str(speaker_wav_path), 16000)
            logger.info(f"[{req_id}] Loaded reference speaker WAV for prompt_speech_16k, shape: {prompt_speech_16k_np.shape}")
        except Exception as e_load_wav:
            logger.error(f"[{req_id}] Failed to load reference speaker WAV {speaker_wav_path} with cosyvoice.utils.load_wav: {e_load_wav}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid reference speaker audio file: {e_load_wav}")

        if prompt_speech_16k_np.shape[0] / 16000 > 30: # Check duration (assuming mono after load_wav)
             logger.warning(f"[{req_id}] Reference audio duration > 30s ({prompt_speech_16k_np.shape[0] / 16000:.2f}s). CosyVoice might truncate or error.")
             # Consider truncating it here if necessary, or rely on CosyVoice internal handling.
             # For now, let CosyVoice handle it, but be aware of its assertion.

        # Determine which inference method to use based on language
        # For cross-lingual (ref lang != target lang), inference_cross_lingual is often preferred.
        # For same-lingual, inference_zero_shot is fine.
        # CosyVoice `inference_zero_shot` also claims cross-lingual capabilities.
        # Let's stick to `inference_zero_shot` as it's simpler and more general for cloning.
        
        # The `language` parameter for XTTS was for the output. CosyVoice infers from text
        # or uses language tags in text for some models/methods.
        # For `inference_zero_shot`, the `text_to_synthesize` should be in the target language.
        # The `prompt_speech_16k` provides the voice.
        
        logger.info(f"[{req_id}] Calling inference_zero_shot. Output SR will be {cosy_model.sample_rate}Hz.")
        
        # The iterator yields chunks if stream=True, or one chunk if stream=False
        generated_chunks = []
        for i, result_chunk in enumerate(cosy_model.inference_zero_shot(
            text=text_to_synthesize, # Text in target language
            prompt_text=style_prompt_text if style_prompt_text else "", # Optional style prompt
            prompt_speech_16k=prompt_speech_16k_np, # Reference speaker audio (numpy array)
            stream=False # Get full audio at once for now
        )):
            generated_chunks.append(result_chunk['tts_speech']) # torch tensor
            logger.info(f"[{req_id}] Received chunk {i}, len: {result_chunk['tts_speech'].shape[1]/cosy_model.sample_rate:.2f}s")
        
        if not generated_chunks:
            logger.error(f"[{req_id}] CosyVoice inference returned no audio chunks.")
            raise HTTPException(status_code=500, detail="Speech synthesis produced no audio.")

        # Concatenate if multiple chunks (though stream=False should give one)
        full_speech_tensor = torch.cat(generated_chunks, dim=1)

        output_wav_path = temp_dir / f"cosyvoice_output_{req_id}.wav"
        torchaudio.save(str(output_wav_path), full_speech_tensor.cpu(), cosy_model.sample_rate)
        
        if not output_wav_path.exists() or output_wav_path.stat().st_size < 100: # Basic check
            logger.error(f"[{req_id}] Synthesis produced invalid output file: {output_wav_path}")
            raise HTTPException(status_code=500, detail="Synthesis produced invalid output file.")

        logger.info(f"[{req_id}] Synthesis successful! Output: {output_wav_path} (Size: {output_wav_path.stat().st_size}, SR: {cosy_model.sample_rate}Hz)")
        
        # Stream the file and cleanup
        async def file_iterator_with_cleanup(file_path: Path, dir_to_clean: str) -> Generator[bytes, None, None]:
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536): # Read in 64KB chunks
                        yield chunk
            finally:
                await _cleanup_temp_dir_async(dir_to_clean)

        return StreamingResponse(
            file_iterator_with_cleanup(output_wav_path, temp_dir_path_str),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=cosyvoice_cloned_{req_id}.wav"}
        )

    except HTTPException: # Re-raise HTTPExceptions
        await _cleanup_temp_dir_async(temp_dir_path_str)
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error during speech generation: {e}", exc_info=True)
        await _cleanup_temp_dir_async(temp_dir_path_str)
        raise HTTPException(status_code=500, detail=f"Error during speech generation: {str(e)}")