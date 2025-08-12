# Docker/cosyvoice_api.py - v6 (Definitive Fix: In-Memory Streaming)
import sys
import os
import torch
import torchaudio
from pathlib import Path
import tempfile
import shutil
import logging
import uuid
import io # Import the in-memory I/O library

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# We need StreamingResponse, not FileResponse
from fastapi.responses import StreamingResponse
from typing import Optional, Dict

# Add CosyVoice to the Python path
sys.path.insert(0, '/app/CosyVoice')
sys.path.append('/app/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice API v3 (Multi-Model)", description="API for TTS with voice cloning using specific models per language.")

# --- Model Management ---
# Use a dictionary to hold loaded models to avoid reloading
loaded_models: Dict[str, CosyVoice2] = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths for your models inside the Docker container
# IMPORTANT: These paths must match where you put them in your Dockerfile
MODEL_PATHS = {
    "default": "/app/models/CosyVoice2-0.5B",  # For ja, ko, zh, etc.
    "greek": "/app/models/CosyVoice-Greek"         # For the new Greek model
}

def get_model(language_code: str) -> Optional[CosyVoice2]:
    """Loads a model into memory if not already loaded, based on the language code."""
    # Decide which model key to use. 'el' is the code for Greek.
    model_key = "greek" if language_code == 'el' else "default"
    
    if model_key in loaded_models:
        logger.info(f"Using cached model for key: '{model_key}'")
        return loaded_models[model_key]

    model_path_str = MODEL_PATHS.get(model_key)
    if not model_path_str:
        logger.error(f"No model path configured for model key '{model_key}'")
        return None
    
    model_path = Path(model_path_str)
    if not model_path.exists():
        logger.error(f"Model file does not exist for key '{model_key}': {model_path}")
        return None

    logger.info(f"Loading model for '{model_key}' from path: {model_path}")
    try:
        model = CosyVoice2(str(model_path))
        loaded_models[model_key] = model
        logger.info(f"SUCCESS: CosyVoice model '{model_path.name}' loaded. Sample rate: {model.sample_rate}")
        return model
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading model for '{model_key}': {e}", exc_info=True)
        return None

@app.on_event("startup")
def startup_event():
    """
    On startup, we just log the device info and pre-warm the default model.
    """
    logger.info(f"PyTorch environment detected device: {device}")
    logger.info("CosyVoice API started. Models will be loaded on demand.")
    logger.info("Pre-warming the default model...")
    # This will load the main model into memory so it's ready.
    get_model('default')

@app.get("/health")
def health_check():
    """
    Health check is healthy if the default model is either already
    loaded or can be successfully loaded on demand.
    """
    if "default" in loaded_models:
        return {"status": "healthy", "message": "Default CosyVoice model is loaded and ready."}
    
    if get_model('default'):
        return {"status": "healthy", "message": "CosyVoice default model is available and loads successfully."}
    else:
        return {"status": "unhealthy", "message": "FATAL: CosyVoice default model could not be loaded."}

@app.post("/generate-speech/")
async def generate_speech(
    text_to_synthesize: str = Form(...),
    reference_speaker_wav: UploadFile = File(...),
    # This new field is crucial for selecting the right model
    target_language_code: str = Form(...), # e.g., 'ja', 'ko', 'zh', 'el'
    style_prompt_text: Optional[str] = Form("")
):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] /generate-speech called for file '{reference_speaker_wav.filename}'.")

    # Dynamically get the correct model for the requested language
    cosy_model = get_model(target_language_code)

    if not cosy_model:
        raise HTTPException(status_code=503, detail=f"TTS Service not available: A model for the language '{target_language_code}' could not be loaded.")
        
    # We still need a temp directory for the UPLOADED reference file
    with tempfile.TemporaryDirectory(prefix=f"cosy_api_ref_{req_id}_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        speaker_wav_path = temp_dir_path / "reference.wav"
        try:
            with open(speaker_wav_path, "wb") as f:
                shutil.copyfileobj(reference_speaker_wav.file, f)
        finally:
            reference_speaker_wav.file.close()

        try:
            prompt_speech_16k = load_wav(str(speaker_wav_path), 16000)
            if prompt_speech_16k is None:
                raise ValueError("load_wav returned None. The audio file may be invalid or empty.")
            
            output = cosy_model.inference_zero_shot(text_to_synthesize, style_prompt_text, prompt_speech_16k)
            result_chunk = next(output)
            
            if 'tts_speech' not in result_chunk or result_chunk['tts_speech'] is None:
                raise HTTPException(status_code=500, detail="Speech synthesis produced no valid audio.")

            tts_speech_tensor = result_chunk['tts_speech']
            
            # --- THE DEFINITIVE FIX ---
            # 1. Create an in-memory binary buffer.
            buffer = io.BytesIO()
            
            # 2. Save the audio tensor directly into the buffer in WAV format.
            torchaudio.save(buffer, tts_speech_tensor.cpu(), cosy_model.sample_rate, format="wav")
            
            # 3. Rewind the buffer to the beginning so it can be read.
            buffer.seek(0)
            
            # 4. Return a StreamingResponse that reads from the in-memory buffer.
            # This avoids creating/deleting any temporary files for the output.
            return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=generated_speech.wav"})
            # --- END OF FIX ---

        except Exception as e:
            logger.error(f"[{req_id}] An error occurred during synthesis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal error occurred in the TTS engine: {e}")