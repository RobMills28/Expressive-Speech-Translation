# Docker/api_inference_logic.py (v13 - "In-Place" Execution)
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
from tqdm import tqdm

# Imports now work directly because the WORKDIR is correct.
from musetalk.utils.utils import load_all_model, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.audio_processor import AudioProcessor
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.structures import merge_data_samples
from transformers import WhisperModel

logger = logging.getLogger(__name__)

# This helper function is fine, no changes needed.
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
    if MODELS: return True

    logger.info("Loading all models for the API...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # This function now correctly finds its models because the WORKDIR is /app/MuseTalk
    vae, unet, pe = load_all_model(device=device)
    MODELS['vae'] = vae
    MODELS['unet'] = unet
    MODELS['pe'] = pe

    # Paths are now relative to the current WORKDIR (/app/MuseTalk)
    models_dir = Path("./models")
    
    pose_config = str(Path('./musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'))
    pose_checkpoint = str(models_dir / 'dwpose/dw-ll_ucoco_384.pth')
    MODELS['pose_estimator'] = init_pose_estimator(pose_config, pose_checkpoint, device=str(device))

    whisper_feature_extractor_path = str(models_dir / "whisper")
    
    # --- THIS IS THE CORRECTED/RE-INSERTED LINE ---
    MODELS['audio_processor'] = AudioProcessor(feature_extractor_path=whisper_feature_extractor_path)

    from musetalk.utils.face_parsing import FaceParsing
    MODELS['face_parser'] = FaceParsing()
    
    MODELS['whisper_model'] = WhisperModel.from_pretrained(whisper_feature_extractor_path).to(device=device).eval()

    logger.info("All models loaded successfully.")
    return True

def run_lip_sync(video_path_str: str, audio_path_str: str, bbox_shift: int) -> str:
    device = torch.device("cpu")
    input_video_path = Path(video_path_str)
    temp_dir = input_video_path.parent
    
    logger.info("Step 1: Extracting frames from video...")
    frames_dir = temp_dir / "input_frames"
    frames_dir.mkdir()
    cmd = f"ffmpeg -v fatal -i {video_path_str} -start_number 0 -q:v 2 {frames_dir}/%08d.png"
    subprocess.run(cmd, shell=True, check=True)
    
    img_list = sorted([str(p) for p in frames_dir.glob('*.png')])
    if not img_list: raise ValueError("FFmpeg failed to extract frames.")

    logger.info(f"Step 2: Preprocessing {len(img_list)} frames...")
    pose_estimator_model = MODELS['pose_estimator']
    coords_list, original_frames = get_landmark_and_bbox(img_list, upperbondrange=bbox_shift)
    
    if not coords_list: raise ValueError("Failed to get bboxes from video frames.")

    fps = get_video_fps(video_path_str)
    # smooth_bbox is a good utility, let's keep it
    coords_list = smooth_bbox(coords_list, window_size=5)

    audio_processor = MODELS['audio_processor']
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path_str)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, MODELS['unet'].model.dtype, MODELS['whisper_model'], librosa_length, fps=fps)

    input_latent_list = []
    for bbox, frame in zip(coords_list, original_frames):
        if bbox == coord_placeholder:
            input_latent_list.append(None)
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        if crop_frame.size == 0:
            input_latent_list.append(None)
            continue
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = MODELS['vae'].get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    logger.info("Step 3: Starting inference loop...")
    generated_frames_dir = temp_dir / "generated_frames"
    generated_frames_dir.mkdir()
    
    timesteps = torch.tensor([0], device=device)
    unet = MODELS['unet'].model
    
    vae_wrapper = MODELS['vae']
    pe = MODELS['pe']
    fp = MODELS['face_parser']
    
    # Use our corrected datagen
    gen = datagen(whisper_chunks, input_latent_list, batch_size=8)
    
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, desc="Generating frames")):
        if latent_batch is None: continue
        
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(device=device, dtype=unet.dtype)
        
        pred_latents = unet(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        
        recon = vae_wrapper.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    logger.info("Step 4: Blending frames...")
    # Pad the original_frames and coords_list to match the length of res_frame_list
    # This can happen if audio is longer than video
    num_generated_frames = len(res_frame_list)
    while len(original_frames) < num_generated_frames:
        original_frames.append(original_frames[-1])
        coords_list.append(coords_list[-1])

    for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending frames")):
        bbox = coords_list[i]
        if bbox == coord_placeholder: 
            # If no face, just use the original frame
            cv2.imwrite(str(generated_frames_dir / f"{i:08d}.png"), original_frames[i])
            continue
        
        ori_frame = original_frames[i]
        x1, y1, x2, y2 = bbox
        
        try:
            res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
        except Exception:
            cv2.imwrite(str(generated_frames_dir / f"{i:08d}.png"), ori_frame)
            continue
            
        combine_frame = get_image(ori_frame, res_frame_resized, bbox, mode="jaw", fp=fp)
        cv2.imwrite(str(generated_frames_dir / f"{i:08d}.png"), combine_frame)

    logger.info("Step 5: Stitching frames into video...")
    temp_silent_video_path = temp_dir / "temp_silent.mp4"
    final_output_video_path = temp_dir / "final_output.mp4"

    ffmpeg_cmd_frames_to_video = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(generated_frames_dir / "%08d.png"), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(temp_silent_video_path)]
    subprocess.run(ffmpeg_cmd_frames_to_video, check=True, capture_output=True, text=True)
    
    ffmpeg_cmd_mux_audio = ["ffmpeg", "-y", "-i", str(temp_silent_video_path), "-i", audio_path_str, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(final_output_video_path)]
    subprocess.run(ffmpeg_cmd_mux_audio, check=True, capture_output=True, text=True)
    
    return str(final_output_video_path)