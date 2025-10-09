import logging
import numpy as np
import librosa
import cv2
import tempfile
from pathlib import Path
from speechbrain.inference import SpeakerRecognition
from deepface import DeepFace

logger = logging.getLogger(__name__)

# Global model instances (loaded once)
_speaker_model = None
_ser_pipeline = None

def get_speaker_model(device="cpu"):
    """Lazy load speaker recognition model"""
    global _speaker_model
    if _speaker_model is None:
        logger.info("Loading SpeechBrain speaker recognition model...")
        _speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
    return _speaker_model

def calculate_speaker_similarity(source_audio_path: Path, translated_audio_path: Path, device="cpu") -> float:
    """Calculate speaker similarity using your existing function"""
    logger.info(f"Calculating Speaker Similarity for {translated_audio_path.name}...")
    try:
        spkrec_model = get_speaker_model(device)
        score, _ = spkrec_model.verify_files(str(source_audio_path), str(translated_audio_path))
        return float(score.squeeze())
    except Exception as e:
        logger.error(f"Speaker similarity calculation failed: {e}")
        return 0.0

def calculate_acoustic_features(audio_path: Path) -> dict:
    """Extract acoustic features using your existing function"""
    logger.info(f"Calculating Acoustic Features for {audio_path.name}...")
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        
        # Your existing acoustic feature extraction
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if np.any(f0) else 0.0
        f0_std = np.nanstd(f0) if np.any(f0) else 0.0

        rms_energy = librosa.feature.rms(y=y)
        intensity_mean = np.mean(rms_energy)
        intensity_std = np.std(rms_energy)
        
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-6)

        # Calculate RMS and Peak
        rms_mean = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))

        return {
            "f0_mean": float(f0_mean),
            "f0_std": float(f0_std),
            "intensity_mean": float(intensity_mean),
            "intensity_std": float(intensity_std),
            "hnr": float(hnr),
            "rms_mean": float(rms_mean),
            "peak_amplitude": float(peak_amplitude)
        }
    except Exception as e:
        logger.error(f"Failed to calculate acoustic features: {e}")
        return {"f0_mean": 0, "f0_std": 0, "intensity_mean": 0, "intensity_std": 0, "hnr": 0}

def get_audio_emotion(audio_path: Path) -> str:
    """
    Analyze emotion from audio. This is a placeholder implementation.
    I could integrate with emotion recognition models here.
    """
    try:
        # Load audio and calculate basic emotional indicators
        y, sr = librosa.load(str(audio_path), sr=16000)
        
        if len(y) == 0:
            return "neutral"
        
        # Simple heuristic based on acoustic features
        # This is a placeholder - I could use more sophisticated models
        
        # Calculate energy and pitch variation as emotion indicators
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_f0 = f0[~np.isnan(f0)]
        pitch_variation = np.std(voiced_f0) if len(voiced_f0) > 0 else 0.0
        
        # Simple classification based on energy and pitch variation
        if energy > 0.02 and pitch_variation > 20:
            emotion = "excited"
        elif energy < 0.01:
            emotion = "calm"
        elif pitch_variation > 15:
            emotion = "expressive"
        else:
            emotion = "neutral"
        
        logger.debug(f"Audio emotion for {audio_path.name}: {emotion} (energy={energy:.4f}, pitch_var={pitch_variation:.2f})")
        return emotion
        
    except Exception as e:
        logger.error(f"Error analyzing audio emotion for {audio_path}: {e}", exc_info=True)
        return "neutral"

def get_visual_emotion(video_path: Path, temp_dir: Path) -> str:
    """Extract visual emotion using your existing function"""
    logger.info(f"Classifying visual emotion for {video_path.name}...")
    try:
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

        temp_frame_path = temp_dir / f"{video_path.stem}_mid_frame.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        analysis = DeepFace.analyze(
            img_path=str(temp_frame_path), 
            actions=['emotion'], 
            enforce_detection=True,
            silent=True
        )
        
        if analysis and isinstance(analysis, list):
            return analysis[0]['dominant_emotion']
        return "no_face_detected"
        
    except Exception as e:
        logger.warning(f"Visual emotion classification failed: {e}")
        return "error_or_no_face"