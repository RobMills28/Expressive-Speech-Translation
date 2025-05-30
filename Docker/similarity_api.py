# Docker/similarity_api.py
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import logging
import os

# Import your VoiceSimilarityAnalyzer
# Assuming voice_similarity_analyzer.py is in the same /app directory in the container
from voice_similarity_analyzer import VoiceSimilarityAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Similarity API", description="API for comparing voice similarity of audio files.")

# Preload the model on startup
@app.on_event("startup")
async def startup_event():
    try:
        VoiceSimilarityAnalyzer._load_model() # Ensure model is loaded when service starts
    except Exception as e:
        logger.error(f"Failed to preload speaker embedding model on startup: {e}", exc_info=True)
        # The application can still start, get_speaker_embedding will try to load it again.

@app.post("/compare-voices/")
async def compare_voices_endpoint(
    original_audio: UploadFile = File(..., description="Original/reference audio file."),
    cloned_audio: UploadFile = File(..., description="Cloned audio file.")
):
    request_id = str(os.urandom(4).hex()) # Short unique ID for the request
    logger.info(f"[{request_id}] Received request for /compare-voices/")
    logger.info(f"[{request_id}] Original audio: {original_audio.filename}, Cloned audio: {cloned_audio.filename}")

    with tempfile.TemporaryDirectory(prefix=f"similarity_{request_id}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        original_audio_path = temp_dir / original_audio.filename
        cloned_audio_path = temp_dir / cloned_audio.filename

        try:
            # Save uploaded files to the temporary directory
            with open(original_audio_path, "wb") as f_orig:
                shutil.copyfileobj(original_audio.file, f_orig)
            logger.info(f"[{request_id}] Saved original audio to {original_audio_path}")

            with open(cloned_audio_path, "wb") as f_cloned:
                shutil.copyfileobj(cloned_audio.file, f_cloned)
            logger.info(f"[{request_id}] Saved cloned audio to {cloned_audio_path}")

        except Exception as e_save:
            logger.error(f"[{request_id}] Error saving uploaded files: {e_save}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error saving uploaded audio files.")
        finally:
            original_audio.file.close()
            cloned_audio.file.close()

        try:
            similarity_score = VoiceSimilarityAnalyzer.compare_audio_files(
                str(original_audio_path),
                str(cloned_audio_path)
            )

            if similarity_score is None:
                logger.error(f"[{request_id}] Failed to calculate similarity score.")
                raise HTTPException(status_code=500, detail="Failed to calculate similarity score.")

            logger.info(f"[{request_id}] Similarity score: {similarity_score:.4f}")
            return JSONResponse(content={"similarity_score": similarity_score, "status": "success"})

        except Exception as e:
            logger.error(f"[{request_id}] Error during similarity comparison: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during similarity comparison: {str(e)}")

@app.get("/health")
async def health_check():
    # Basic health check, can be expanded to check model loading status
    try:
        if VoiceSimilarityAnalyzer._model is None:
             VoiceSimilarityAnalyzer._load_model() # Attempt to load if not already
        if VoiceSimilarityAnalyzer._model is not None:
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "unhealthy", "model_loaded": False, "reason": "Speaker embedding model not loaded"}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "reason": str(e)}

if __name__ == "__main__":
    # This part is for local testing of the API if needed, Docker CMD will override.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)