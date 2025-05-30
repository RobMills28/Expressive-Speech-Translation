# services/voice_similarity_analyzer.py
import torch
import torchaudio
import numpy as np
# from speechbrain.pretrained import EncoderClassifier # Deprecated, see UserWarning
from speechbrain.inference.speaker import EncoderClassifier # Corrected import path
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import os # Added for os.getenv
from typing import Optional # <--- ADDED THIS IMPORT

logger = logging.getLogger(__name__)

class VoiceSimilarityAnalyzer:
    _model = None
    _device = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            try:
                logger.info("Loading speaker embedding model (ECAPA-TDNN)...")
                # Use environment variable for pretrained path if set, otherwise default
                savedir_base = Path(os.getenv("SPEECHBRAIN_PRETRAINED_PATH", "pretrained_models")) # Path in container
                model_savedir = savedir_base / "spkrec-ecapa-voxceleb" # Model specific sub-directory
                # No need to mkdir here if SPEECHBRAIN_PRETRAINED_PATH in Dockerfile creates /app/pretrained_models
                # SpeechBrain will create the model_savedir subdirectory itself.
                
                cls._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(model_savedir), 
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                cls._model.eval() 
                cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Speaker embedding model loaded successfully on {cls._device}.")
            except Exception as e:
                logger.error(f"Failed to load speaker embedding model: {e}", exc_info=True)
                cls._model = None 
                raise 
        return cls._model

    @classmethod
    def get_speaker_embedding(cls, audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
        try:
            model = cls._load_model()
            if model is None: return None
            logger.debug(f"Loading audio for embedding: {audio_path}")
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            
            # Normalize waveform to [-1, 1] if not already (SpeechBrain expects this)
            if waveform.abs().max() > 1.0: # A simple check, more robust normalization might be needed
                waveform = waveform / waveform.abs().max()
            
            waveform = waveform.to(cls._device)
            with torch.no_grad():
                # Ensure wav_lens is correctly shaped and on the same device
                wav_lens_tensor = torch.tensor([waveform.shape[1]], device=cls._device).float() / waveform.shape[1]
                embedding = model.encode_batch(waveform, wav_lens=wav_lens_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            logger.debug(f"Successfully extracted embedding for {audio_path}, shape: {embedding.shape}")
            return embedding
        except FileNotFoundError: logger.error(f"Audio file not found for embedding: {audio_path}"); return None
        except Exception as e: logger.error(f"Error extracting speaker embedding for {audio_path}: {e}", exc_info=True); return None

    @classmethod
    def calculate_similarity(cls, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> Optional[float]:
        if embedding1 is None or embedding2 is None:
            logger.warning("Cannot calculate similarity, one or both embeddings are None."); return None
        try:
            if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            logger.debug(f"Calculated cosine similarity: {similarity:.4f}")
            return float(similarity)
        except Exception as e: logger.error(f"Error calculating cosine similarity: {e}", exc_info=True); return None

    @classmethod
    def compare_audio_files(cls, original_audio_path: str, cloned_audio_path: str) -> Optional[float]:
        logger.info(f"Comparing voice similarity between '{Path(original_audio_path).name}' and '{Path(cloned_audio_path).name}'")
        original_embedding = cls.get_speaker_embedding(original_audio_path)
        if original_embedding is None: logger.error(f"Could not get embedding for original: {original_audio_path}"); return None
        cloned_embedding = cls.get_speaker_embedding(cloned_audio_path)
        if cloned_embedding is None: logger.error(f"Could not get embedding for cloned: {cloned_audio_path}"); return None
        similarity_score = cls.calculate_similarity(original_embedding, cloned_embedding)
        if similarity_score is not None: logger.info(f"Voice similarity score: {similarity_score:.4f}")
        else: logger.warning("Failed to calculate voice similarity score.")
        return similarity_score