# analyze_outputs.py
import argparse
import pandas as pd
import logging
from pathlib import Path
import subprocess
import tempfile
import random  # For dummy data generation
import json
import numpy as np
import cv2
import librosa

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.stats import pearsonr


# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def run_external_command(command: list, job_id: str) -> bool:
    """A robust helper to run external command-line tools like OpenFace."""
    try:
        logging.info(f"[{job_id}] Running command: {' '.join(command)}")
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10-minute timeout for heavy tools
        )
        logging.debug(f"[{job_id}] STDOUT: {process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"[{job_id}] Command failed. STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"[{job_id}] Command timed out.")
        return False
    except FileNotFoundError:
        logging.error(f"[{job_id}] Command not found: {command[0]}. Is the tool installed and in the PATH?")
        return False

def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extracts audio from a video file using ffmpeg."""
    audio_path = output_dir / f"{video_path.stem}_extracted.wav"
    command = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        str(audio_path)
    ]
    if run_external_command(command, video_path.stem):
        return audio_path
    raise IOError(f"Failed to extract audio for {video_path}")

def load_transcripts(transcript_path: Path) -> dict:
    """
    Loads source and target transcripts from a JSON file.
    Returns a dictionary with default values on failure.
    """
    if not transcript_path.is_file():
        logging.warning(f"  [WARN] Transcript file not found at {transcript_path}. Using default text.")
        return {"source_text": "text not found", "target_text": "text not found"}
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                "source_text": data.get("source_text", "key missing"),
                "target_text": data.get("target_text", "key missing")
            }
    except Exception as e:
        logging.error(f"  [ERROR] Could not read or parse {transcript_path}: {e}")
        return {"source_text": "file read error", "target_text": "file read error"}

# ==============================================================================
# METRIC CALCULATION FUNCTIONS (PLACEHOLDERS)
# NOTE: Fill these in with real model inference code on the HPC.
# ==============================================================================

# --- LINGUISTIC METRICS ---

def calculate_linguistic_scores(source_text: str, target_text: str, sonar_model) -> dict:
    """Calculates SONAR and BERTScore."""
    logging.info("  [REAL] Calculating SONAR and BERTScore...")
    try:
        # Calculate BERTScore
        P, R, F1 = bert_score_calc([target_text], [source_text], lang="en", rescale_with_baseline=True, verbose=False)
        bert_f1 = F1.mean().item()

        # Calculate SONAR score
        sonar_score_val = sonar_model.score([source_text], [target_text], batch_size=4)['score']
        
        return {
            "sonar_score": sonar_score_val,
            "bert_score_f1": bert_f1,
        }
    except Exception as e:
        logging.error(f"  [ERROR] Linguistic score calculation failed: {e}")
        return {"sonar_score": 0.0, "bert_score_f1": 0.0}

# --- AUDIO METRICS ---

def calculate_speaker_similarity(source_audio: Path, translated_audio: Path, spkrec_model) -> float:
    """Calculates speaker similarity using a pre-trained x-vector model."""
    logging.info(f"  [REAL] Calculating Speaker Similarity for {translated_audio.name}...")
    try:
        score, _ = spkrec_model.verify_files(str(source_audio), str(translated_audio))
        return float(score.squeeze())
    except Exception as e:
        logging.error(f"  [ERROR] Speaker similarity calculation failed: {e}")
        return 0.0

# --- NEW: SPEECH EMOTION RECOGNITION ---

def get_audio_emotion(audio_path: Path, ser_pipeline) -> str:
    """
    Classifies emotion from audio using a pre-loaded Hugging Face pipeline.
    """
    logging.info(f"  [REAL] Classifying audio emotion for {audio_path.name}...")
    try:
        predictions = ser_pipeline(str(audio_path), top_k=1)
        # The pipeline returns a list of lists, so we access the first element of each
        if predictions and predictions[0]:
            return predictions[0][0]['label']
        return "no_prediction"
    except Exception as e:
        logging.error(f"  [ERROR] Audio emotion classification failed for {audio_path.name}: {e}")
        return "error"


def calculate_acoustic_features(audio_path: Path) -> dict:
    """
    Extracts a suite of low-level acoustic features using Librosa.
    This provides a quantitative measure of audio naturalness and quality.
    """
    logging.info(f"  [REAL] Calculating Acoustic Features for {audio_path.name}...")
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        
        # Pitch (Fundamental Frequency - F0)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if np.any(f0) else 0.0
        f0_std = np.nanstd(f0) if np.any(f0) else 0.0

        # Intensity (Root Mean Square Energy)
        rms_energy = librosa.feature.rms(y=y)
        intensity_mean = np.mean(rms_energy)
        intensity_std = np.std(rms_energy)
        
        # Harmonics-to-Noise Ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-6)

        return {
            "f0_mean": float(f0_mean),
            "f0_std": float(f0_std),
            "intensity_mean": float(intensity_mean),
            "intensity_std": float(intensity_std),
            "hnr": float(hnr)
        }
    except Exception as e:
        logging.error(f"  [ERROR] Failed to calculate acoustic features for {audio_path.name}: {e}")
        return {"f0_mean": 0, "f0_std": 0, "intensity_mean": 0, "intensity_std": 0, "hnr": 0}

# --- VISUAL METRICS ---

def get_visual_emotion(video_path: Path, temp_dir: Path) -> str:
    """
    Extracts the dominant emotion from a central frame of a video using DeepFace.
    """
    logging.info(f"  [REAL] Classifying visual emotion for {video_path.name}...")
    try:
        # 1. Open the video and find the middle frame to avoid start/end credits
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError("Cannot open video file.")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_index = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Cannot read the middle frame.")

        # 2. Save the frame to a temporary image file
        temp_frame_path = temp_dir / f"{video_path.stem}_mid_frame.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        # 3. Analyze the frame with DeepFace
        analysis = DeepFace.analyze(
            img_path=str(temp_frame_path), 
            actions=['emotion'], 
            enforce_detection=True,
            silent=True # Suppress verbose console output
        )
        
        # DeepFace returns a list of dicts, one for each face found
        if analysis and isinstance(analysis, list):
            return analysis[0]['dominant_emotion']
        return "no_face_detected"
        
    except Exception as e:
        # This will catch errors from OpenCV or if DeepFace finds no face
        logging.warning(f"  [WARN] Visual emotion classification failed for {video_path.name}: {e}")
        return "error_or_no_face"

def calculate_av_sync_confidence(video: Path, audio: Path, avsync_model) -> float:
    """Calculates Audio-Visual Sync Confidence using a pre-trained model."""
    logging.info(f"  [REAL] Calculating AV-Sync Confidence for {video.name}...")
    try:
        # The pyi-sad library provides a direct way to get this score
        confidence_score = avsync_model.get_sync_confidence(str(video))
        return float(confidence_score)
    except Exception as e:
        logging.error(f"  [ERROR] AV-Sync confidence calculation failed: {e}")
        return 0.0

def calculate_visual_identity_similarity(source_video_path: Path, translated_video_path: Path, temp_dir: Path) -> float:
    """Calculates facial identity similarity using DeepFace."""
    logging.info(f"  [REAL] Calculating Visual Identity Sim for {translated_video_path.name}...")
    try:
        # Extract a middle frame from each to compare
        cap_source = cv2.VideoCapture(str(source_video_path))
        source_frame_idx = int(cap_source.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        cap_source.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
        ret1, frame1 = cap_source.read()
        cap_source.release()
        
        cap_trans = cv2.VideoCapture(str(translated_video_path))
        trans_frame_idx = int(cap_trans.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        cap_trans.set(cv2.CAP_PROP_POS_FRAMES, trans_frame_idx)
        ret2, frame2 = cap_trans.read()
        cap_trans.release()

        if not ret1 or not ret2:
            raise ValueError("Could not read frames from one or both videos.")
        
        result = DeepFace.verify(frame1, frame2, model_name="ArcFace", enforce_detection=False)
        return float(result['distance'])
    except Exception as e:
        logging.error(f"  [ERROR] Visual identity similarity failed: {e}")
        return 0.0

# --- DETAILED FACIAL ANALYSIS ---

def run_openface_analysis(video: Path, output_dir: Path) -> Path:
    """Runs the OpenFace FeatureExtraction tool and returns the path to the output CSV."""
    # NOTE: Assumes OpenFace is installed and `FeatureExtraction` is in the system PATH.
    output_csv_path = output_dir / f"{video.stem}_openface.csv"
    command = [
        'FeatureExtraction',
        '-f', str(video),
        '-out_dir', str(output_dir),
        '-aus' # Extract Action Units
    ]
    if run_external_command(command, video.stem):
        # OpenFace renames the output file, so we find it and rename it back.
        generated_file = output_dir / f"{video.stem.replace(' ', '_')}.csv"
        if generated_file.exists():
            generated_file.rename(output_csv_path)
            return output_csv_path
    raise RuntimeError(f"OpenFace analysis failed for {video}")

def run_mediapipe_analysis(video_path: Path, audio_path: Path, landmarker) -> dict:
    """
    Extracts detailed facial features and lip-sync correlation using MediaPipe.

    Args:
        video_path (Path): Path to the video file.
        audio_path (Path): Path to the corresponding audio file.
        landmarker (vision.FaceLandmarker): The pre-loaded MediaPipe FaceLandmarker model.

    Returns:
        dict: A dictionary of calculated metrics.
    """
    logging.info(f"  [REAL] Running MediaPipe analysis for {video_path.name}...")
    try:
        # --- 1. Extract facial features from video frame by frame ---
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: raise ValueError("Video FPS is zero, cannot process.")

        mouth_openings = []
        head_yaws = []
        head_pitches = []

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to MediaPipe's RGB Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get timestamp for the current frame
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Detect face landmarks
            face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if face_landmarker_result.face_landmarks:
                # Get landmarks for the first detected face
                landmarks = face_landmarker_result.face_landmarks[0]
                
                # Key landmarks for mouth opening: top lip (13) and bottom lip (14)
                p_top = landmarks[13]
                p_bottom = landmarks[14]
                
                # Calculate Euclidean distance in 2D (x, y)
                mouth_opening = np.sqrt((p_top.x - p_bottom.x)**2 + (p_top.y - p_bottom.y)**2)
                mouth_openings.append(mouth_opening)

                # Extract head pose from the transformation matrix
                if face_landmarker_result.facial_transformation_matrixes:
                    matrix = face_landmarker_result.facial_transformation_matrixes[0]
                    # Simple pitch/yaw extraction (more complex methods exist)
                    pitch = np.arcsin(-matrix[1][2])
                    yaw = np.arctan2(matrix[0][2], matrix[2][2])
                    head_pitches.append(pitch)
                    head_yaws.append(yaw)
            else:
                # If no face is detected, append a neutral value
                mouth_openings.append(0.0)
                head_pitches.append(0.0)
                head_yaws.append(0.0)
            
            frame_index += 1
        
        cap.release()

        # --- 2. Extract audio envelope, synchronised to the video frame rate ---
        y, sr = librosa.load(str(audio_path), sr=16000)
        frame_length_in_samples = int(sr / fps)
        
        audio_envelope = np.array([
            np.mean(np.abs(y[i : i + frame_length_in_samples]))
            for i in range(0, len(y), frame_length_in_samples)
        ])

        # --- 3. Correlate the two signals ---
        min_len = min(len(mouth_openings), len(audio_envelope))
        if min_len > 1: # Pearson correlation requires at least 2 data points
            correlation, _ = pearsonr(mouth_openings[:min_len], audio_envelope[:min_len])
        else:
            correlation = 0.0

        # --- 4. Calculate final statistics ---
        return {
            "head_pose_pitch_std": float(np.std(head_pitches)),
            "head_pose_yaw_std": float(np.std(head_yaws)),
            "lip_audio_correlation": float(correlation)
        }

    except Exception as e:
        logging.error(f"  [ERROR] MediaPipe analysis failed for {video_path.name}: {e}", exc_info=True)
        return {"head_pose_pitch_std": 0, "head_pose_yaw_std": 0, "lip_audio_correlation": 0}

def run_deepfake_detection(video: Path, dfdc_repo_path: Path, temp_dir: Path) -> float:
    """
    Runs the selimsef/dfdc_deepfake_challenge detector on a single video.
    This assumes the Docker image for the detector has been built.
    
    Args:
        video (Path): Path to the video file to analyze.
        dfdc_repo_path (Path): Path to the cloned DFDC repository.
        temp_dir (Path): A temporary directory to store the output CSV.

    Returns:
        float: The predicted deepfake probability score (lower is better).
    """
    job_id = video.stem
    logging.info(f"  [{job_id}] Running Deepfake Detection...")

    # The detector script expects a FOLDER of videos, not a single file.
    # To handle this, we create a temporary input folder for just this one video.
    video_input_folder = temp_dir / "dfdc_input"
    video_input_folder.mkdir(exist_ok=True)
    
    # Create a symbolic link to the video to avoid copying large files.
    (video_input_folder / video.name).symlink_to(video.resolve())

    output_csv = temp_dir / "submission.csv"
    
    # This command runs the pre-built Docker image named 'df'.
    # It mounts the necessary directories:
    # - The video folder is mounted as /videos inside the container.
    # - The temp output folder is mounted as /output.
    # - The pre-trained model weights are mounted to the expected cache location.
    docker_command = [
        "docker", "run", "--runtime=nvidia", "--ipc=host", "--rm",
        "-v", f"{str(video_input_folder)}:/videos",
        "-v", f"{str(temp_dir)}:/output",
        "-v", f"{str(dfdc_repo_path / 'weights')}:/root/.cache/torch/checkpoints",
        "df",  # This is the name of the Docker image from the README
        "/bin/bash", "-c",
        # This is the command that runs inside the container.
        "./predict_submission.sh /videos /output/submission.csv"
    ]

    if run_external_command(docker_command, job_id):
        try:
            # If successful, parse the single-line output CSV.
            predictions = pd.read_csv(output_csv)
            if not predictions.empty:
                score = predictions.iloc[0]['label']
                logging.info(f"  [{job_id}] Deepfake score: {score:.4f}")
                return float(score)
        except Exception as e:
            logging.error(f"  [{job_id}] Could not parse DFDC output file: {e}")
    
    # On any failure, return 1.0 (the worst possible score).
    return 1.0

# ==============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# ==============================================================================

def analyze_single_job(job_info: dict, mcf_dir: Path, seamless_dir: Path, analysis_temp_dir: Path, dfdc_repo_path: Path, ser_pipeline) -> dict:
    """
    Analyzes all outputs for a single job from the manifest, now with all metrics.
    """
    job_id = job_info['job_id']
    source_video_path = Path(job_info['source_video_path'])
    logging.info(f"--- Analyzing Job ID: {job_id} ({source_video_path.name}) ---")

    # Define expected paths for processed files
    mcf_video_path = mcf_dir / f"{job_id}.mp4"
    seamless_video_path = seamless_dir / f"{job_id}.mp4"

    if not mcf_video_path.exists() or not seamless_video_path.exists():
        logging.error(f"Output video(s) not found for job {job_id}. Skipping.")
        return None

    # --- Create a temporary directory for this job's intermediate files ---
    job_temp_dir = analysis_temp_dir / job_id
    job_temp_dir.mkdir(exist_ok=True)

    # --- Extract Audio (once per source) ---
    source_audio_path = extract_audio(source_video_path, job_temp_dir)
    mcf_audio_path = extract_audio(mcf_video_path, job_temp_dir)
    seamless_audio_path = extract_audio(seamless_video_path, job_temp_dir)

    # --- Load Transcripts ---
    # Defines the expected path for the JSON transcript files that your
    # batch processing pipeline should create alongside each video.
    mcf_transcript_path = mcf_dir / f"{job_id}_transcripts.json"
    seamless_transcript_path = seamless_dir / f"{job_id}_transcripts.json"
    
    # Calls the robust helper function to load the transcript data from the files.
    mcf_transcripts = load_transcripts(mcf_transcript_path)
    seamless_transcripts = load_transcripts(seamless_transcript_path)

    # Initialize the results dictionary that will store all metrics for this job.
    results = {"job_id": job_id, "dataset": job_info['dataset'], "source_video": source_video_path.name}
    
    # Store the actual text in the results file for later qualitative review.
    results['mcf_source_text'] = mcf_transcripts['source_text']
    results['mcf_target_text'] = mcf_transcripts['target_text']
    results['seamless_source_text'] = seamless_transcripts['source_text']
    results['seamless_target_text'] = seamless_transcripts['target_text']

    # --- Source Video/Audio Analysis (Baseline) ---
    logging.info("Analyzing SOURCE for baseline metrics...")
    results.update({f"source_{k}": v for k, v in calculate_acoustic_features(source_audio_path).items()})
    results['source_audio_emotion'] = get_audio_emotion(source_audio_path, models['ser'])
    results['source_visual_emotion'] = get_visual_emotion(source_video_path, job_temp_dir)
    results['source_deepfake_score'] = run_deepfake_detection(source_video_path, dfdc_repo_path, job_temp_dir)
    results.update({f"source_{k}": v for k, v in run_mediapi_equation(source_video_path, source_audio_path, models['face_landmarker']).items()})
    results['source_openface_output'] = str(run_openface_analysis(source_video_path, job_temp_dir))

    # --- MCF Analysis ---
    logging.info("Analyzing Modern Cascaded Framework (MCF) output...")
    results.update({f"mcf_{k}": v for k, v in calculate_linguistic_scores(mcf_transcripts['source_text'], mcf_transcripts['target_text'], models['sonar']).items()})
    results.update({f"mcf_{k}": v for k, v in calculate_acoustic_features(mcf_audio_path).items()})
    results['mcf_audio_emotion'] = get_audio_emotion(mcf_audio_path, models['ser'])
    results['mcf_speaker_sim'] = calculate_speaker_similarity(source_audio_path, mcf_audio_path, models['spkrec'])
    results['mcf_visual_emotion'] = get_visual_emotion(mcf_video_path, job_temp_dir)
    results['mcf_av_sync'] = calculate_av_sync_confidence(mcf_video_path, mcf_audio_path, models['avsync'])
    results['mcf_identity_sim'] = calculate_visual_identity_similarity(source_video_path, mcf_video_path, job_temp_dir)
    results['mcf_deepfake_score'] = run_deepfake_detection(mcf_video_path, dfdc_repo_path, job_temp_dir)
    results.update({f"mcf_{k}": v for k, v in run_mediapipe_analysis(mcf_video_path, mcf_audio_path, models['face_landmarker']).items()})
    results['mcf_openface_output'] = str(run_openface_analysis(mcf_video_path, job_temp_dir))
    
    # --- Seamless Expressive Analysis ---
    logging.info("Analyzing Seamless Expressive output...")
    results.update({f"seamless_{k}": v for k, v in calculate_linguistic_scores(seamless_transcripts['source_text'], seamless_transcripts['target_text'], models['sonar']).items()})
    results.update({f"seamless_{k}": v for k, v in calculate_acoustic_features(seamless_audio_path).items()})
    results['seamless_audio_emotion'] = get_audio_emotion(seamless_audio_path, models['ser'])
    results['seamless_speaker_sim'] = calculate_speaker_similarity(source_audio_path, seamless_audio_path, models['spkrec'])
    results['seamless_visual_emotion'] = get_visual_emotion(seamless_video_path, job_temp_dir)
    results['seamless_av_sync'] = calculate_av_sync_confidence(seamless_video_path, seamless_audio_path, models['avsync'])
    results['seamless_identity_sim'] = calculate_visual_identity_similarity(source_video_path, seamless_video_path, job_temp_dir)
    results['seamless_deepfake_score'] = run_deepfake_detection(seamless_video_path, dfdc_repo_path, job_temp_dir)
    results.update({f"seamless_{k}": v for k, v in run_mediapipe_analysis(seamless_video_path, seamless_audio_path, models['face_landmarker']).items()})
    results['seamless_openface_output'] = str(run_openface_analysis(seamless_video_path, job_temp_dir))

    return results-

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extensive batch analysis of multimodal translation outputs.")
    parser.add_argument("manifest_file", type=Path, help="Path to the batch_manifest.csv file.")
    parser.add_argument("mcf_output_dir", type=Path, help="Directory containing MCF output videos and data.")
    parser.add_argument("seamless_output_dir", type=Path, help="Directory containing Seamless output videos and data.")
    parser.add_argument("--output_csv", type=Path, default=Path("analysis_results.csv"), help="Path to save the final results CSV.")
    parser.add_argument("--dfdc_repo", type=Path, required=True, help="Path to the cloned selimsef/dfdc_deepfake_challenge repository.")
    
    # NEW: Argument to specify the SER model. Default is a standard public model.
    parser.add_argument("--ser_model", type=str, default="superb/wav2vec2-base-superb-er", help="Hugging Face model ID for Speech Emotion Recognition.")
    
    args = parser.parse_args()

    if not args.manifest_file.is_file():
        logging.critical(f"Error: Manifest file not found at {args.manifest_file}")
        return

    # --- Load ALL Models ONCE at the start for efficiency ---
    logging.info("Loading all analysis models...")
    try:
        # Create MediaPipe FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task') # YOU MUST DOWNLOAD THIS FILE
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        
        # A single dictionary to hold all pre-loaded models
        models = {
            "ser": pipeline("audio-classification", model=args.ser_model, device=0),
            "spkrec": SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"}),
            "sonar": Sonar(),
            "avsync": SAD(device='cuda'),
            "face_landmarker": vision.FaceLandmarker.create_from_options(options)
        }
        logging.info("All models loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load a required model. Aborting. Error: {e}", exc_info=True)
        return

    if not args.manifest_file.is_file():
        logging.critical(f"Error: Manifest file not found at {args.manifest_file}")
        return

    # --- Load Models ONCE at the start for efficiency ---
    logging.info(f"Loading Speech Emotion Recognition model: {args.ser_model}...")
    try:
        # Load the pipeline onto the first available GPU (device=0)
        ser_pipeline = pipeline("audio-classification", model=args.ser_model, device=0)
        logging.info("SER model loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load SER model. Aborting. Error: {e}")
        return

    # Create a temporary directory for all analysis artifacts
    with tempfile.TemporaryDirectory(prefix="analysis_") as temp_dir_str:
        analysis_temp_dir = Path(temp_dir_str)
        logging.info(f"Created temporary directory for analysis artifacts: {analysis_temp_dir}")
        
        manifest_df = pd.read_csv(args.manifest_file)
        all_results = []

        for _, job_info in manifest_df.iterrows():
            try:
                # Pass the loaded SER pipeline to the analysis function for use in the loop
                job_results = analyze_single_job(job_info, args.mcf_output_dir, args.seamless_output_dir, analysis_temp_dir, args.dfdc_repo, ser_pipeline)
                if job_results:
                    all_results.append(job_results)
            except Exception as e:
                logging.critical(f"CRITICAL ERROR: Failed to process job {job_info['job_id']}. Skipping. Error: {e}", exc_info=True)
                continue
            
        if not all_results:
            logging.warning("No jobs were successfully analyzed. No output file will be created.")
            return

        # Save all results to a final CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output_csv, index=False)
        logging.info(f"\n[SUCCESS] Analysis complete. Results for {len(all_results)} jobs saved to '{args.output_csv}'")