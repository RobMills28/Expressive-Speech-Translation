# services/voice_similarity_analyzer.py
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier # Corrected import path
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import os
from typing import Optional

logger = logging.getLogger(__name__)

class VoiceSimilarityAnalyzer:
    _model = None
    _device = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            try:
                logger.info("Loading speaker embedding model (ECAPA-TDNN)...")
                savedir_base = Path(os.getenv("SPEECHBRAIN_PRETRAINED_PATH", "pretrained_models"))
                model_savedir = savedir_base / "spkrec-ecapa-voxceleb"
                
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

            logger.debug(f"Loaded waveform for {Path(audio_path).name}: shape={waveform.shape}, sr={sr}, dtype={waveform.dtype}, min={waveform.min():.4f}, max={waveform.max():.4f}, mean={waveform.mean():.4f}")

            if waveform.shape[0] > 1: # Ensure mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.debug(f"Converted to mono: shape={waveform.shape}")
            if sr != target_sr:
                logger.debug(f"Resampling from {sr}Hz to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                logger.debug(f"Resampled waveform: shape={waveform.shape}, sr={target_sr}")

            # Normalize waveform amplitude to [-1, 1] - SpeechBrain ECAPA-TDNN expects this range.
            # This is a crucial step.
            current_max_abs = waveform.abs().max()
            if current_max_abs > 0: # Avoid division by zero for silent audio
                 if current_max_abs > 1.0: # Only normalize if it exceeds the range
                    waveform = waveform / current_max_abs
                    logger.debug(f"Normalized waveform amplitude (was > 1.0): new min={waveform.min():.4f}, max={waveform.max():.4f}")
                 else:
                    logger.debug(f"Waveform amplitude already within [-1,1] (max_abs={current_max_abs:.4f}). No normalization needed.")
            else:
                logger.warning(f"Waveform for {Path(audio_path).name} is silent or near silent (max_abs={current_max_abs:.4f}). Embedding may be poor.")


            waveform = waveform.to(cls._device)
            with torch.no_grad():
                # wav_lens expects values between 0 and 1.
                wav_lens_tensor = torch.tensor([waveform.shape[1]], device=cls._device).float() / waveform.shape[1]
                embedding = model.encode_batch(waveform, wav_lens=wav_lens_tensor)
                embedding = embedding.squeeze().cpu().numpy() # Squeeze to (192,)
            logger.debug(f"Successfully extracted embedding for {Path(audio_path).name}, shape: {embedding.shape}")
            return embedding
        except FileNotFoundError: logger.error(f"Audio file not found for embedding: {audio_path}"); return None
        except Exception as e: logger.error(f"Error extracting speaker embedding for {audio_path}: {e}", exc_info=True); return None

    @classmethod
    def calculate_similarity(cls, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> Optional[float]:
        if embedding1 is None or embedding2 is None:
            logger.warning("Cannot calculate similarity, one or both embeddings are None."); return None
        try:
            # Ensure embeddings are 2D for cosine_similarity: (1, D)
            if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
            
            if embedding1.shape[1] != embedding2.shape[1]:
                logger.error(f"Embedding dimensions mismatch: {embedding1.shape} vs {embedding2.shape}")
                return None

            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            logger.debug(f"Calculated cosine similarity: {similarity:.4f}")
            return float(similarity)
        except Exception as e: logger.error(f"Error calculating cosine similarity: {e}", exc_info=True); return None

    @classmethod
    def compare_audio_files(cls, original_audio_path: str, cloned_audio_path: str) -> Optional[float]:
        logger.info(f"Comparing voice similarity between '{Path(original_audio_path).name}' and '{Path(cloned_audio_path).name}'")
        original_embedding = cls.get_speaker_embedding(original_audio_path)
        if original_embedding is None: logger.error(f"Could not get embedding for original: {Path(original_audio_path).name}"); return None
        
        cloned_embedding = cls.get_speaker_embedding(cloned_audio_path)
        if cloned_embedding is None: logger.error(f"Could not get embedding for cloned: {Path(cloned_audio_path).name}"); return None
        
        similarity_score = cls.calculate_similarity(original_embedding, cloned_embedding)
        if similarity_score is not None: logger.info(f"Voice similarity score: {similarity_score:.4f}")
        else: logger.warning("Failed to calculate voice similarity score.")
        return similarity_score