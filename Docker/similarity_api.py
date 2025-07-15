# Docker/similarity_api.py
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import logging
import os
import uuid # For more unique request IDs

# Import your VoiceSimilarityAnalyser
from voice_similarity_analyser import VoiceSimilarityAnalyser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Similarity API", description="API for comparing voice similarity of audio files.")

@app.on_event("startup")
async def startup_event():
    try:
        VoiceSimilarityAnalyser._load_model()
        logger.info("Speaker embedding model preloaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to preload speaker embedding model on startup: {e}", exc_info=True)

@app.post("/compare-voices/")
async def compare_voices_endpoint(
    original_audio: UploadFile = File(..., description="Original/reference audio file."),
    cloned_audio: UploadFile = File(..., description="Cloned audio file.")
):
    request_id = str(uuid.uuid4())[:8] # Generate a unique ID for each request
    logger.info(f"[{request_id}] Received request for /compare-voices/")
    logger.info(f"[{request_id}] Original audio: {original_audio.filename} (type: {original_audio.content_type}), Cloned audio: {cloned_audio.filename} (type: {cloned_audio.content_type})")

    with tempfile.TemporaryDirectory(prefix=f"similarity_{request_id}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        # Use secure and unique filenames in temp storage
        original_audio_path = temp_dir / f"original_{request_id}{Path(original_audio.filename).suffix}"
        cloned_audio_path = temp_dir / f"cloned_{request_id}{Path(cloned_audio.filename).suffix}"

        try:
            with open(original_audio_path, "wb") as f_orig:
                shutil.copyfileobj(original_audio.file, f_orig)
            logger.info(f"[{request_id}] Saved original audio to {original_audio_path} (size: {original_audio_path.stat().st_size} bytes)")

            with open(cloned_audio_path, "wb") as f_cloned:
                shutil.copyfileobj(cloned_audio.file, f_cloned)
            logger.info(f"[{request_id}] Saved cloned audio to {cloned_audio_path} (size: {cloned_audio_path.stat().st_size} bytes)")

        except Exception as e_save:
            logger.error(f"[{request_id}] Error saving uploaded files: {e_save}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error saving uploaded audio files.")
        finally:
            if original_audio.file: original_audio.file.close()
            if cloned_audio.file: cloned_audio.file.close()

        try:
            similarity_score = VoiceSimilarityAnalyser.compare_audio_files(
                str(original_audio_path),
                str(cloned_audio_path)
            )

            if similarity_score is None:
                logger.error(f"[{request_id}] Failed to calculate similarity score (compare_audio_files returned None).")
                raise HTTPException(status_code=500, detail="Failed to calculate similarity score due to internal error in analyser.")

            logger.info(f"[{request_id}] Similarity score: {similarity_score:.4f}")
            return JSONResponse(content={"similarity_score": similarity_score, "request_id": request_id, "status": "success"})

        except Exception as e: # Catch-all for other unexpected errors during comparison
            logger.error(f"[{request_id}] Error during similarity comparison logic: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during similarity comparison: {str(e)}")

@app.get("/health")
async def health_check():
    model_loaded_flag = False
    reason = "Speaker embedding model not loaded or error during check."
    try:
        if VoiceSimilarityAnalyser._model is None:
             VoiceSimilarityAnalyser._load_model() 
        if VoiceSimilarityAnalyser._model is not None:
            model_loaded_flag = True
            reason = "Speaker embedding model loaded."
            return {"status": "healthy", "model_loaded": True, "message": reason}
        else: # Model is still None after attempting load
             return {"status": "unhealthy", "model_loaded": False, "message": "Failed to load speaker embedding model."}
    except Exception as e:
        logger.error(f"Health check model loading exception: {e}", exc_info=True)
        return {"status": "unhealthy", "model_loaded": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)