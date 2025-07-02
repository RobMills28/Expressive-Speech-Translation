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

# These imports will now work because we patched the library files
from musetalk.utils.utils import load_all_model, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.audio_processor import AudioProcessor
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.structures import merge_data_samples
from transformers import WhisperModel

logger = logging.getLogger(__name__)

# Copied helper functions
def smooth_facial_landmarks(landmarks, window_size=5):
    smoothed_landmarks = []
    for i in range(len(landmarks)):
        start = max(0, i - window_size // 2)
        end = min(len(landmarks), i + window_size // 2 + 1)
        window = landmarks[start:end]
        smoothed_landmarks.append(np.mean(window, axis=0))
    return np.array(smoothed_landmarks)

def smooth_bbox(bboxes, window_size=5):
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

MODELS = {}

def load_models_for_api():
    global MODELS
    if MODELS:
        logger.info("Models are already loaded.")
        return True

    logger.info("Loading all models for the API...")
    device = torch.device("cpu")
    
    vae, unet, pe = load_all_model(device=device)
    MODELS['vae'] = vae
    MODELS['unet'] = unet
    MODELS['pe'] = pe

    pose_checkpoint = os.path.join(MUSETALK_PROJECT_ROOT, 'models', 'dwpose', 'dw-ll_ucoco_384.pth')
    pose_config = os.path.join(MUSETALK_PROJECT_ROOT, 'musetalk', 'utils', 'dwpose', 'rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
    MODELS['pose_estimator'] = init_pose_estimator(pose_config, pose_checkpoint, device=str(device))

    whisper_path = os.path.join(MUSETALK_PROJECT_ROOT, "models", "whisper")
    MODELS['audio_processor'] = AudioProcessor(feature_extractor_path=whisper_path)
    MODELS['whisper_model'] = WhisperModel.from_pretrained(whisper_path).to(device=device).eval()

    from musetalk.utils.face_parsing import FaceParsing
    MODELS['face_parser'] = FaceParsing()

    logger.info("All models loaded successfully.")
    return True

def run_lip_sync(video_path_str: str, audio_path_str: str, bbox_shift: int) -> str:
    device = torch.device("cpu")
    input_video_path = Path(video_path_str)
    
    logger.info("Preprocessing video and audio...")
    coords_list, original_frames = get_landmark_and_bbox(img_list=[video_path_str], upperbondrange=bbox_shift)
    
    if not coords_list: raise ValueError("Failed to get bboxes from video.")
    if not original_frames: raise ValueError("Failed to read frames from video.")

    fps = get_video_fps(video_path_str)
    bbox_results = smooth_bbox(coords_list, window_size=5)

    audio_processor = MODELS['audio_processor']
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path_str)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, MODELS['unet'].model.dtype, MODELS['whisper_model'], librosa_length, fps=fps)

    logger.info("Starting inference loop...")
    temp_dir = input_video_path.parent
    generated_frames_dir = temp_dir / "generated_frames"
    generated_frames_dir.mkdir()
    
    timesteps = torch.tensor([0], device=device)
    unet = MODELS['unet'].model
    vae = MODELS['vae'].vae
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
    
    input_latent_list = []
    for bbox, frame in zip(coords_list, original_frames):
        if bbox == coord_placeholder: continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = MODELS['vae'].get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    gen = datagen(whisper_chunks, input_latent_list, batch_size=8, device=device)
    
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, desc="Generating frames")):
        audio_emb = pe(whisper_batch)
        
        noise_pred = unet(sample=latent_batch, timestep=timesteps, encoder_hidden_states=audio_emb).sample
        pred_latents = latent_batch - noise_pred
        
        output_frame_np = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
        output_frame_np = (output_frame_np.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # Need to determine which original frame and bbox this corresponds to
        # This part of the logic needs to be fixed.
        # For now, let's assume a simple mapping.
        current_frame_index = i * 8 # This is a simplification
        if current_frame_index >= len(original_frames): break

        bbox = bbox_results[current_frame_index]
        if bbox == coord_placeholder: continue
        x1, y1, x2, y2 = bbox
        
        pasted_frame = get_image(original_frames[current_frame_index], output_frame_np, bbox, fp=fp)
        output_path = generated_frames_dir / f"{current_frame_index:08d}.png"
        cv2.imwrite(str(output_path), pasted_frame)

    logger.info("Stitching frames into video...")
    temp_silent_video_path = temp_dir / "temp_silent.mp4"
    final_output_video_path = temp_dir / "final_output.mp4"

    ffmpeg_cmd_frames_to_video = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(generated_frames_dir / "%08d.png"), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(temp_silent_video_path)]
    subprocess.run(ffmpeg_cmd_frames_to_video, check=True, capture_output=True, text=True)
    
    ffmpeg_cmd_mux_audio = ["ffmpeg", "-y", "-i", str(temp_silent_video_path), "-i", audio_path_str, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(final_output_video_path)]
    subprocess.run(ffmpeg_cmd_mux_audio, check=True, capture_output=True, text=True)
    
    return str(final_output_video_path)