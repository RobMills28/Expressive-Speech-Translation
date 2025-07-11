import os
from pathlib import Path
import shutil
import tempfile
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

# This import will now succeed because we have fixed the path above.
from api_inference_logic import load_models_for_api, run_lip_sync

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d (%(funcName)s)] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MuseTalk LipSync API", description="API for lip-syncing videos using MuseTalk.")

@app.on_event("startup")
def startup_event():
    # Load all models once when the server starts up
    try:
        load_models_for_api()
        logger.info("API startup complete. Models loaded.")
    except Exception as e:
        logger.critical(f"API startup failed during model loading: {e}", exc_info=True)
        # The app will likely fail to start properly, but this log is crucial
        
@app.get("/health")
def health_check():
    # A simple health check can just confirm that the API is running
    return {"status": "healthy", "message": "MuseTalk API is running. Check startup logs for model status."}

def _save_upload_file(upload_file: UploadFile, temp_dir: Path) -> Path:
    try:
        # Use a safe suffix, default to .tmp if none
        suffix = Path(upload_file.filename if upload_file.filename else ".tmp").suffix
        temp_file_path = temp_dir / f"{uuid.uuid4()}{suffix}"
        with temp_file_path.open("wb") as f_buffer:
            shutil.copyfileobj(upload_file.file, f_buffer)
        return temp_file_path
    finally:
        if hasattr(upload_file, 'file') and upload_file.file:
            upload_file.file.close()

@app.post("/lipsync-video/")
async def lipsync_video_endpoint(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    bbox_shift: int = Form(0)
):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"--- [{req_id}] /lipsync-video CALLED ---")
    
    temp_dir_path = Path(tempfile.mkdtemp(prefix=f"musetalk_api_{req_id}_"))
    try:
        input_video_path = _save_upload_file(video_file, temp_dir_path)
        input_audio_path = _save_upload_file(audio_file, temp_dir_path)

        # Call the lip-sync logic
        final_video_path_str = run_lip_sync(str(input_video_path), str(input_audio_path), bbox_shift)
        
        final_video_path = Path(final_video_path_str)
        if not final_video_path.exists():
            raise HTTPException(status_code=500, detail="Processing failed to produce an output video.")
            
        return FileResponse(final_video_path, media_type="video/mp4", filename=f"musetalk_lipsynced_{req_id}.mp4")

    except Exception as e:
        logger.error(f"[{req_id}] Error during lip-sync process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # This cleanup is important to prevent filling up the disk
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)