# Magenta AI/Docker/musetalk_api.py
import sys
import os
import shutil
import tempfile
import subprocess
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any # Added List

import torch
import numpy as np # Added numpy
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse 

# Add MuseTalk to Python path (assuming API is in /app/ and MuseTalk code is in /app/MuseTalk/)
MUSETALK_BASE_PATH = Path("/app/MuseTalk") # This is where MuseTalk code will be in Docker
sys.path.insert(0, str(MUSETALK_BASE_PATH.resolve())) # Add project root
sys.path.insert(0, str(MUSETALK_BASE_PATH / "musetalk")) # Add musetalk module itself

# MuseTalk specific imports (ensure these paths are correct relative to MUSETALK_BASE_PATH)
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet2DConditionModel
from musetalk.models.whisper_encoder import WhisperEncoder # For audio features
# from musetalk.models.face_control_encoder import FaceControlEncoder # May not be loaded separately
# from musetalk.models.sync_encoder import SyncEncoder # May not be loaded separately
from musetalk.pipelines.pipeline_musetalk import MuseTalkPipeline # The main pipeline

# MMPose and MMDet for face/pose processing (these come from your Conda/pip installs)
from mmpose.apis import inference_topdown, init_model as init_pose_estimator # Renamed for clarity
# from mmpose.structures import PoseDataSample # Used by topdown inference
from mmdet.apis import inference_detector, init_detector

# MuseTalk's own preprocessing utils
from musetalk.utils.preprocessing import get_landmark_and_bbox_from_video_paths, read_imgs_from_video_path, smooth_facial_landmarks, smooth_bbox
from musetalk.utils.utils import get_video_fps, get_audio_features_from_path, get_pose_from_audio_features_from_path, load_all_models # Updated utils

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d (%(funcName)s)] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MuseTalk LipSync API", description="API for lip-syncing videos using MuseTalk.")

# --- Global Model Variables & Paths ---
MODELS_DIR = MUSETALK_BASE_PATH / "models"
DEVICE = torch.device("cpu") # For CPU execution in Docker

# Models to be loaded
musetalk_models: Dict[str, Any] = {} # To store all loaded models
pose_estimator: Optional[Any] = None
face_detector: Optional[Any] = None
pipeline: Optional[MuseTalkPipeline] = None

@app.on_event("startup")
async def startup_event():
    global musetalk_models, pose_estimator, face_detector, pipeline
    logger.info("MuseTalk API: Starting up and loading models...")

    # Define model paths (these should match where they are in the Docker image)
    model_paths = {
        "vae": str(MODELS_DIR / "sd-vae"),
        "whisper": str(MODELS_DIR / "whisper"),
        "unet": str(MODELS_DIR / "musetalkV15"), # Using v1.5
        "face_control_encoder": str(MODELS_DIR / "face-parse-bisent"), # Check if this is correct model for face_control_encoder
        "sync_encoder": str(MODELS_DIR / "syncnet") # Check if this is correct model for sync_encoder
    }
    
    # DWPose (MMPose)
    pose_config_file = str(MUSETALK_BASE_PATH / 'configs/dwpose/dwpose-l_384x288.py')
    pose_checkpoint_file = str(MODELS_DIR / 'dwpose/dw-ll_ucoco_384.pth')

    # MMDet (Face Detector) - MuseTalk's get_landmark_and_bbox might use its own or an internal one.
    # If you need a separate MMDet model:
    # det_config_file = str(MUSETALK_BASE_PATH / "configs/mm_utils/faster_rcnn_r50_fpn_1x_coco.py") # Example
    # det_checkpoint_file = str(MODELS_DIR / "mm_utils/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth") # Example

    try:
        logger.info("Loading DWPose model...")
        pose_estimator = init_pose_estimator(pose_config_file, pose_checkpoint_file, device=str(DEVICE))
        logger.info("DWPose model loaded.")

        # logger.info("Loading Face Detector model (MMDet)...")
        # face_detector = init_detector(det_config_file, det_checkpoint_file, device=str(DEVICE))
        # logger.info("Face Detector model loaded.")
        # NOTE: MuseTalk's `get_landmark_and_bbox_from_video_paths` seems to handle face detection internally, possibly using dlib or another method.
        # If so, a separate MMDet model might not be needed here if relying on their util.

        logger.info("Loading MuseTalk core models (VAE, UNet, Whisper, etc.)...")
        musetalk_models = load_all_models(model_paths, device=DEVICE) # load_all_models is from musetalk.utils.utils
        
        # Create the pipeline
        pipeline = MuseTalkPipeline(
            vae=musetalk_models["vae"],
            unet=musetalk_models["unet"],
            whisper_encoder=musetalk_models["whisper_encoder"],
            face_control_encoder=musetalk_models["face_control_encoder"],
            sync_encoder=musetalk_models["sync_encoder"],
            device=DEVICE
        )
        logger.info("MuseTalkPipeline initialized.")
        logger.info("MuseTalk API: All models loaded successfully.")

    except Exception as e:
        logger.error(f"FATAL: Error loading MuseTalk models during startup: {e}", exc_info=True)
        # Set all to None to indicate failure for health check
        musetalk_models = {}
        pose_estimator = face_detector = pipeline = None

@app.get("/health")
async def health_check():
    if pipeline and pose_estimator and musetalk_models.get("vae") and musetalk_models.get("unet"): # Check key components
        return {"status": "healthy", "message": "MuseTalk API is running and models appear loaded."}
    else:
        logger.error("/health: MuseTalk models not ready.")
        return {"status": "unhealthy", "message": "MuseTalk models not loaded or failed to load."}, 503

def _save_upload_file(upload_file: UploadFile, temp_dir: Path) -> Path:
    try:
        # Use a generic suffix or try to get from filename, default to .tmp
        suffix = Path(upload_file.filename if upload_file.filename else "input").suffix or ".tmp"
        temp_file_path = temp_dir / f"{uuid.uuid4()}{suffix}"
        with open(temp_file_path, "wb") as f_buffer:
            shutil.copyfileobj(upload_file.file, f_buffer)
        logger.info(f"Saved uploaded file '{upload_file.filename}' to '{temp_file_path}' (size: {temp_file_path.stat().st_size if temp_file_path.exists() else 'N/A'})")
        return temp_file_path
    finally:
        if hasattr(upload_file, 'file') and upload_file.file:
            upload_file.file.close()

@app.post("/lipsync-video/")
async def lipsync_video_endpoint(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    # Add other MuseTalk parameters as Form fields if needed, e.g., bbox_shift
    # bbox_shift: int = Form(0) 
):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"--- [{req_id}] /lipsync-video CALLED ---")
    logger.info(f"[{req_id}] Input video: {video_file.filename}, Input audio: {audio_file.filename}")

    if not pipeline or not pose_estimator or not musetalk_models:
        logger.error(f"[{req_id}] MuseTalk models not loaded. Cannot process.")
        raise HTTPException(status_code=503, detail="LipSync Service not available: Models not loaded.")

    temp_dir_path = Path(tempfile.mkdtemp(prefix=f"musetalk_api_{req_id}_"))
    logger.info(f"[{req_id}] Created temporary directory: {temp_dir_path}")

    ffmpeg_path = "ffmpeg" # Assumed to be in PATH in Docker

    try:
        input_video_path = _save_upload_file(video_file, temp_dir_path)
        input_audio_path = _save_upload_file(audio_file, temp_dir_path)

        # --- MUSELTALK CORE PROCESSING LOGIC ---
        # This section needs careful adaptation from MuseTalk's scripts/inference.py
        logger.info(f"[{req_id}] Starting MuseTalk processing for video: {input_video_path}, audio: {input_audio_path}")

        # 1. Video Preprocessing (Frames, Landmarks, BBoxes, Pose)
        # Their script has `get_landmark_and_bbox_from_video_paths` which calls `read_imgs_from_video_path`
        # and then uses `pose_estimator` for pose.
        
        video_fps = get_video_fps(str(input_video_path))
        logger.info(f"[{req_id}] Input video FPS: {video_fps}")
        if video_fps == 0: video_fps = 25.0 # Default if detection fails

        # This function from their utils seems to do a lot of the video prep
        # It internally calls read_imgs, pose_estimator, smoothing etc.
        # It expects a LIST of video paths, so we pass a list with one item.
        # It also expects `args` object for parameters like `bbox_shift`.
        # We'll need to create a dummy args or pass parameters directly.
        
        # Create a simple args-like object or dict for parameters
        class InferenceArgs:
            bbox_shift = [0] # Default, can be overridden by Form input
            exp_coeffs_path = None # Not typically used for basic inference like this
            output_mash = False # Not typically used for basic inference
            mouth_width_ratio = 0.5 # Default from their code
            mouth_height_ratio = 0.5 # Default
            only_mouth = False # Default to full face
            output_original_video = False

        args_for_preprocessing = InferenceArgs()
        # If you add bbox_shift as a Form param: args_for_preprocessing.bbox_shift = [bbox_shift]

        logger.info(f"[{req_id}] Extracting landmarks and bboxes...")
        landmark_results, bbox_results, video_fps_out = get_landmark_and_bbox_from_video_paths(
            [str(input_video_path)], pose_estimator, DEVICE, args_for_preprocessing.bbox_shift
        )
        # video_fps_out might be more accurate than get_video_fps sometimes
        if video_fps_out != 0: video_fps = video_fps_out

        # Smooth landmarks and bboxes (optional but good for quality)
        # landmark_results = [smooth_facial_landmarks(lmk, window_size=5) for lmk in landmark_results]
        # bbox_results = [smooth_bbox(bbox, window_size=5) for bbox in bbox_results]

        logger.info(f"[{req_id}] Landmarks/bboxes extracted for {len(landmark_results[0] if landmark_results else 0)} frames.")

        # 2. Audio Preprocessing (Whisper features, pose from audio)
        logger.info(f"[{req_id}] Extracting audio features...")
        audio_values_16k = read_audio(str(input_audio_path), sample_rate=16000) # From musetalk.utils.utils
        
        # Get audio features and audio-derived pose
        audio_feat_shape_len, audio_feat_torch = get_audio_features_from_path(
            musetalk_models["whisper_encoder"], str(input_audio_path), vad=True
        )
        pose_seq_torch = get_pose_from_audio_features_from_path(audio_feat_torch, audio_feat_shape_len)
        logger.info(f"[{req_id}] Audio features extracted. Audio pose sequence shape: {pose_seq_torch.shape}")

        # 3. MuseTalk Pipeline Generation
        # The pipeline.generate method needs to be called.
        # It typically takes the video frames (or paths), landmarks, bboxes, audio features, pose sequence.
        # Their script processes video in chunks. You might need to adapt that for longer videos.
        # For now, let's assume a single call for a short video.

        generated_frames_dir = temp_dir_path / "musetalk_output_frames"
        generated_frames_dir.mkdir(exist_ok=True)

        logger.info(f"[{req_id}] Running MuseTalkPipeline.generate...")
        # The `generate` method in their pipeline might expect different inputs or structure.
        # This requires careful adaptation of how `scripts/inference.py` calls the pipeline.
        # It expects paths to original frames, not raw RGB data.
        # It also expects paths to landmarks, not raw data.
        # This part is the most complex to adapt correctly.

        # We need to save the preprocessed data (frames, landmarks, bboxes) to disk
        # in a way that the pipeline.generate function or its internal utilities expect.
        # OR modify pipeline.generate / its callers to accept in-memory data.

        # For now, as a placeholder to get the API structure testable, we'll skip actual generation.
        # ** YOU MUST REPLACE THIS WITH THE ACTUAL PIPELINE CALL **
        logger.warning(f"[{req_id}] MUSELTALK PIPELINE GENERATION IS A PLACEHOLDER.")
        # Simulate creating some dummy output frames for ffmpeg to work with
        # This is NOT real MuseTalk output.
        dummy_frame_path = generated_frames_dir / "00000001.png"
        # Create a dummy PNG. You'd need Pillow or opencv-python for this.
        # For now, this will cause ffmpeg to fail if the dummy doesn't exist.
        # Let's assume ffmpeg will fail, and we'll see that in logs.
        # Actual implementation: loop through frames, call pipeline.generate_batch or similar.
        
        # 4. FFmpeg to create video from generated frames and mux with input audio
        temp_silent_video_path = temp_dir_path / f"temp_silent_{req_id}.mp4"
        final_output_video_path = temp_dir_path / f"lipsynced_video_{req_id}.mp4"

        # This assumes frames are saved as %08d.png in generated_frames_dir
        # If no frames are generated by the placeholder above, this ffmpeg call will fail.
        ffmpeg_cmd_frames_to_video = [
            ffmpeg_path, "-y", "-v", "warning",
            "-framerate", str(video_fps), # Use detected video FPS
            "-f", "image2", "-i", str(generated_frames_dir / "%08d.png"),
            "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
            str(temp_silent_video_path)
        ]
        logger.info(f"[{req_id}] Running FFmpeg (frames to video): {' '.join(ffmpeg_cmd_frames_to_video)}")
        try:
            subprocess.run(ffmpeg_cmd_frames_to_video, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"[{req_id}] FFmpeg (frames to video) failed. STDERR: {e.stderr}")
            # If it fails because no frames, copy original video to allow muxing to proceed for testing API flow.
            if "No such file or directory" in e.stderr or "Cannot find a valid frame" in e.stderr:
                 logger.warning(f"[{req_id}] No frames found for ffmpeg, copying original video for muxing test.")
                 shutil.copy(input_video_path, temp_silent_video_path)
            else:
                raise

        ffmpeg_cmd_mux_audio = [
            ffmpeg_path, "-y", "-v", "warning",
            "-i", str(temp_silent_video_path), # Video with lip-sync (or original if frames failed)
            "-i", str(input_audio_path),       # Translated audio from CosyVoice
            "-c:v", "copy",                    # Copy video stream
            "-c:a", "aac", "-b:a", "192k",     # Encode audio to AAC
            "-map", "0:v:0", "-map", "1:a:0",   # Select video from first input, audio from second
            "-shortest",                       # Finish encoding when shortest input ends
            str(final_output_video_path)
        ]
        logger.info(f"[{req_id}] Running FFmpeg (mux audio): {' '.join(ffmpeg_cmd_mux_audio)}")
        subprocess.run(ffmpeg_cmd_mux_audio, check=True,  capture_output=True, text=True)
        # --- END OF MUSELTALK CORE PROCESSING LOGIC (NEEDS FULL IMPLEMENTATION) ---

        if not final_output_video_path.exists() or final_output_video_path.stat().st_size < 1000: # Basic check
            raise HTTPException(status_code=500, detail="Lip-sync process failed to produce a valid output video.")

        logger.info(f"[{req_id}] Lip-sync process completed. Output: {final_output_video_path}")
        
        response = FileResponse(str(final_output_video_path), media_type="video/mp4", filename=f"musetalk_lipsynced_{video_file.filename}")
        # Cleanup should ideally use BackgroundTasks for FileResponse
        # For now, leave temp dir for inspection if something goes wrong. OS will eventually clean /tmp
        # background_tasks.add_task(shutil.rmtree, temp_dir_path)
        return response

    except subprocess.CalledProcessError as e:
        logger.error(f"[{req_id}] FFmpeg command failed. STDERR: {e.stderr} STDOUT: {e.stdout}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr}")
    except HTTPException:
        # If an HTTPException was already raised, just re-raise it
        if temp_dir_path.exists(): shutil.rmtree(temp_dir_path) # Attempt cleanup on known exceptions
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error during lip-sync process: {e}", exc_info=True)
        if temp_dir_path.exists(): shutil.rmtree(temp_dir_path) # Attempt cleanup
        raise HTTPException(status_code=500, detail=f"Unexpected error during lip-sync: {str(e)}")
    # finally: # Not using finally for cleanup with FileResponse without BackgroundTasks
        # if temp_dir_path.exists():
        #     shutil.rmtree(temp_dir_path)
        #     logger.info(f"[{req_id}] Cleaned up temporary directory: {temp_dir_path}")