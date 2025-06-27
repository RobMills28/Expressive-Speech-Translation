import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
import shutil
import subprocess
import logging

# Set up paths and logging
MUSETALK_PROJECT_ROOT = Path("/app/MuseTalk").resolve()
if str(MUSETALK_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MUSETALK_PROJECT_ROOT))
    sys.path.insert(0, str(MUSETALK_PROJECT_ROOT / "musetalk"))

# These imports are now 100% verified against your provided files.
from utils.utils import load_all_model, get_video_fps
from utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from utils.blending import get_image # This one was correct
from utils.audio_processor import AudioProcessor
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.structures import merge_data_samples

logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS COPIED FROM OFFICIAL SCRIPTS ---
# These functions do not exist in the library utility files, so we define them here.
def smooth_facial_landmarks(landmarks, window_size=5):
    """Smooth facial landmarks using a moving average filter."""
    smoothed_landmarks = []
    for i in range(len(landmarks)):
        start = max(0, i - window_size // 2)
        end = min(len(landmarks), i + window_size // 2 + 1)
        window = landmarks[start:end]
        smoothed_landmarks.append(np.mean(window, axis=0))
    return np.array(smoothed_landmarks)

def smooth_bbox(bboxes, window_size=5):
    """Smooth bounding boxes using a moving average filter."""
    smoothed_bboxes = []
    for i in range(len(bboxes)):
        start = max(0, i - window_size // 2)
        end = min(len(bboxes), i + window_size // 2 + 1)
        window = [box for box in bboxes[start:end] if box != coord_placeholder]
        if not window:
            smoothed_bboxes.append(coord_placeholder)
            continue
        avg_box = np.mean(window, axis=0).astype(int)
        smoothed_bboxes.append(tuple(avg_box))
    return smoothed_bboxes
# --- END OF HELPER FUNCTIONS ---

# This dictionary will be populated once at startup
MODELS = {}

def load_models_for_api():
    """
    Loads all necessary models into the global MODELS dictionary.
    This function should be called only once when the API starts.
    """
    global MODELS
    if MODELS:
        logger.info("Models are already loaded.")
        return

    logger.info("Loading all models for the API...")
    device = torch.device("cpu")
    
    vae, unet, pe = load_all_model(device=device)
    MODELS['vae'] = vae
    MODELS['unet'] = unet
    MODELS['pe'] = pe

    models_dir = MUSETALK_PROJECT_ROOT / "models"
    pose_config = str(MUSETALK_PROJECT_ROOT / 'musetalk' / 'utils' / 'dwpose' / 'rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
    pose_checkpoint = str(models_dir / 'dwpose' / 'dw-ll_ucoco_384.pth')
    MODELS['pose_estimator'] = init_pose_estimator(pose_config, pose_checkpoint, device=str(device))

    whisper_path = str(models_dir / "whisper" / "tiny.pt")
    MODELS['audio_processor'] = AudioProcessor(whisper_model_type="tiny", model_path=whisper_path)

    from musetalk.utils.face_parsing.face_parsing import FaceParsing
    MODELS['face_parser'] = FaceParsing()

    logger.info("All models loaded successfully.")
    return True

def run_lip_sync(video_path_str: str, audio_path_str: str, bbox_shift: int) -> str:
    """
    This function contains the core lip-sync logic.
    """
    device = torch.device("cpu")
    input_video_path = Path(video_path_str)
    
    # 1. Preprocessing
    logger.info("Preprocessing video and audio...")
    
    coords_list, original_frames = get_landmark_and_bbox(img_list=[video_path_str], upperbondrange=bbox_shift)
    
    if not coords_list: raise ValueError("Failed to get bboxes from video.")
    if not original_frames: raise ValueError("Failed to read frames from video.")

    fps = get_video_fps(video_path_str)
    bbox_results = smooth_bbox(coords_list, window_size=5)

    audio_processor = MODELS['audio_processor']
    audio_feats = audio_processor.audio2feat(audio_path_str)
    whisper_chunks = audio_processor.feature2chunks(feature_array=audio_feats, fps=fps)

    # 2. Core Inference Loop
    logger.info("Starting inference loop...")
    temp_dir = input_video_path.parent
    generated_frames_dir = temp_dir / "generated_frames"
    generated_frames_dir.mkdir()
    
    timesteps = torch.tensor([0], device=device)
    unet = MODELS['unet']
    vae = MODELS['vae']
    pe = MODELS['pe']
    fp = MODELS['face_parser']

    pose_estimator = MODELS['pose_estimator']
    all_landmarks_for_video = []
    logger.info("Running pose estimation on all frames...")
    for frame in original_frames:
        pose_results = inference_topdown(pose_estimator, frame, bbox_format='xyxy')
        merged_results = merge_data_samples(pose_results)
        keypoints = merged_results.pred_instances.keypoints[0]
        all_landmarks_for_video.append(keypoints)

    lmk_results = smooth_facial_landmarks(all_landmarks_for_video, window_size=5)
    logger.info("Landmark smoothing complete.")

    for i, (bbox, audio_chunk) in enumerate(zip(bbox_results, whisper_chunks)):
        if i >= len(original_frames): break
        if bbox == coord_placeholder: continue
        if i >= len(lmk_results): break # Ensure we don't go out of bounds for landmarks

        lmk = lmk_results[i]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        face_frame_np = original_frames[i][y1:y2, x1:x2]
        
        if face_frame_np.size == 0: continue

        face_frame = Image.fromarray(cv2.cvtColor(face_frame_np, cv2.COLOR_BGR2RGB))
        face_frame = face_frame.resize((256, 256), Image.Resampling.LANCZOS)
        face_frame = np.array(face_frame, dtype=np.float32) / 255.0
        
        latents = vae.encode(torch.from_numpy(face_frame).unsqueeze(0).permute(0, 3, 1, 2).to(device)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        audio_emb = torch.from_numpy(audio_chunk).unsqueeze(0).to(device)
        audio_emb = pe(audio_emb)

        # The inference logic for v1.5 does not seem to use landmarks directly in the UNet
        noise_pred = unet(sample=latents, timestep=timesteps, encoder_hidden_states=audio_emb).sample
        pred_latents = latents - noise_pred
        
        output_frame_np = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
        output_frame_np = (output_frame_np.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        pasted_frame = get_image(original_frames[i], output_frame_np, bbox, fp=fp)
        
        output_path = generated_frames_dir / f"{i:08d}.png"
        cv2.imwrite(str(output_path), pasted_frame)

    # 3. Postprocessing (FFmpeg)
    logger.info("Stitching frames into video...")
    temp_silent_video_path = temp_dir / "temp_silent.mp4"
    final_output_video_path = temp_dir / "final_output.mp4"

    ffmpeg_cmd_frames_to_video = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(generated_frames_dir / "%08d.png"), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(temp_silent_video_path)]
    subprocess.run(ffmpeg_cmd_frames_to_video, check=True, capture_output=True, text=True)
    
    ffmpeg_cmd_mux_audio = ["ffmpeg", "-y", "-i", str(temp_silent_video_path), "-i", audio_path_str, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(final_output_video_path)]
    subprocess.run(ffmpeg_cmd_mux_audio, check=True, capture_output=True, text=True)
    
    return str(final_output_video_path)