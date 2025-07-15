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
from typing import Optional

# Add CosyVoice to the Python path
sys.path.insert(0, '/app/CosyVoice')
sys.path.append('/app/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice API v2", description="API for TTS with voice cloning using CosyVoice2.")

cosy_model: Optional[CosyVoice2] = None
MODEL_PATH = "/app/CosyVoice/pretrained_models/CosyVoice2-0.5B"

@app.on_event("startup")
def startup_event():
    global cosy_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch environment detected device: {device}")
    logger.info(f"Attempting to load CosyVoice2 model from: {MODEL_PATH}")
    
    try:
        cosy_model = CosyVoice2(MODEL_PATH)
        logger.info(f"SUCCESS: CosyVoice model '{Path(MODEL_PATH).name}' loaded. Sample rate: {cosy_model.sample_rate}")
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading CosyVoice model: {e}", exc_info=True)
        cosy_model = None

@app.get("/health")
def health_check():
    if cosy_model:
        return {"status": "healthy", "message": f"CosyVoice model ({Path(MODEL_PATH).name}) loaded."}
    else:
        return {"status": "unhealthy", "message": "CosyVoice model is not loaded."}

@app.post("/generate-speech/")
async def generate_speech(
    text_to_synthesize: str = Form(...),
    reference_speaker_wav: UploadFile = File(...),
    style_prompt_text: Optional[str] = Form("")
):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] /generate-speech called for file '{reference_speaker_wav.filename}'.")

    if not cosy_model:
        raise HTTPException(status_code=503, detail="TTS Service not available: Model not loaded.")

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