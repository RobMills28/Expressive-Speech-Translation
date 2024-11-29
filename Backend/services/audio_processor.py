import os
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from .audio_diagnostics import AudioDiagnostics

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Enhanced audio processor with quality improvements and diagnostics.
    """
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}
    MAX_AUDIO_LENGTH = 300  # seconds (5 minutes)
    SAMPLE_RATE = 16000
    
    # Language-specific processing parameters
    LANGUAGE_PARAMS = {
        'fra': {  # French
            'noise_reduction': 1.1,
            'compression_threshold': 0.35,
            'compression_ratio': 1.6,
            'brightness': 1.1,
            'clarity': 1.2
        },
        'deu': {  # German
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.8,
            'brightness': 0.9,
            'clarity': 1.1
        },
        'spa': {  # Spanish
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.8,
            'brightness': 1.0,
            'clarity': 1.15
        },
        'ita': {  # Italian
            'noise_reduction': 1.1,
            'compression_threshold': 0.3,
            'compression_ratio': 1.5,
            'brightness': 1.05,
            'clarity': 1.25
        },
        'por': {  # Portuguese
            'noise_reduction': 1.3,
            'compression_threshold': 0.45,
            'compression_ratio': 1.7,
            'brightness': 0.95,
            'clarity': 1.2
        }
    }

    def __init__(self):
        """Initialize audio processor with diagnostics capability"""
        try:
            self.diagnostics = AudioDiagnostics()
        except Exception as e:
            logger.error(f"Failed to initialize AudioDiagnostics: {str(e)}")
            self.diagnostics = None

    def validate_audio_length(self, audio_path: str) -> tuple[bool, str]:
        """
        Validates audio file length and basic integrity
        """
        try:
            if Path(audio_path).suffix.lower() not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {self.SUPPORTED_FORMATS}"

            if not os.path.exists(audio_path):
                return False, "Audio file not found"
                
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, "Audio file is empty"

            metadata = torchaudio.info(audio_path)
            
            if metadata.sample_rate <= 0:
                return False, "Invalid sample rate detected"
                
            if metadata.num_frames <= 0:
                return False, "No audio frames detected"

            duration = metadata.num_frames / metadata.sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration <= 0:
                return False, "Invalid audio duration"
                
            if duration > self.MAX_AUDIO_LENGTH:
                return False, f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({self.MAX_AUDIO_LENGTH}s)"

            return True, ""

        except Exception as e:
            error_msg = f"Error validating audio: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def apply_spectral_enhancement(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """Apply language-specific spectral enhancement"""
        try:
            # Get language parameters with fallback to French if language not found
            params = self.LANGUAGE_PARAMS.get(target_language, self.LANGUAGE_PARAMS['fra'])
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Apply STFT for frequency domain processing
            spec = torch.stft(
                audio[0],
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048),
                return_complex=True
            )
            
            freq_bins = spec.shape[1]
            enhance_mask = torch.ones(freq_bins)
            
            # Apply language-specific enhancements
            if target_language == 'fra':  # French needs clearer nasals
                nasal_range = slice(freq_bins // 4, freq_bins // 2)
                enhance_mask[nasal_range] *= params['clarity']
            elif target_language == 'deu':  # German needs softer consonants
                consonant_range = slice(3 * freq_bins // 4, None)
                enhance_mask[consonant_range] *= 0.9
            elif target_language == 'spa':  # Spanish needs clear consonants
                consonant_range = slice(2 * freq_bins // 3, None)
                enhance_mask[consonant_range] *= params['clarity']
            elif target_language == 'ita':  # Italian needs clear vowels
                vowel_range = slice(freq_bins // 3, 2 * freq_bins // 3)
                enhance_mask[vowel_range] *= params['clarity']
            elif target_language == 'por':  # Portuguese needs nasal clarity
                nasal_range = slice(freq_bins // 4, freq_bins // 2)
                enhance_mask[nasal_range] *= params['clarity']
            
            # Apply brightness control
            high_freq_range = slice(3 * freq_bins // 4, None)
            enhance_mask[high_freq_range] *= params['brightness']
            
            # Apply frequency mask
            spec = spec * enhance_mask.unsqueeze(-1)
            
            # Convert back to time domain
            audio = torch.istft(
                spec,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048)
            ).unsqueeze(0)
            
            return audio
            
        except Exception as e:
            logger.error(f"Spectral enhancement failed: {str(e)}")
            return audio

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Process audio file for translation.
        Basic processing without enhancements.
        """
        try:
            logger.info(f"Loading audio from: {audio_path}")
            
            info = torchaudio.info(audio_path)
            logger.info(f"Audio info - Sample rate: {info.sample_rate}, Channels: {info.num_channels}")
            
            audio, orig_freq = torchaudio.load(audio_path)
            
            # Validate audio data
            if torch.isnan(audio).any():
                raise ValueError("Audio contains NaN values")
            if torch.isinf(audio).any():
                raise ValueError("Audio contains infinite values")
            if audio.abs().max() == 0:
                raise ValueError("Audio is silent")
            
            logger.info(f"Original audio shape: {audio.shape}, Frequency: {orig_freq}Hz")
            
            # Process large files in chunks
            if audio.shape[1] > 1_000_000:
                logger.info("Processing large audio in chunks")
                chunks = audio.split(1_000_000, dim=1)
                audio = torch.cat([chunk for chunk in chunks], dim=1)
            
            # Resample if needed
            if orig_freq != self.SAMPLE_RATE:
                logger.info(f"Resampling from {orig_freq}Hz to {self.SAMPLE_RATE}Hz")
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=orig_freq,
                    new_freq=self.SAMPLE_RATE
                )
            
            # Convert to mono
            if audio.shape[0] > 1:
                logger.info("Converting from stereo to mono")
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Normalize
            if audio.abs().max() > 1.0:
                logger.info("Normalizing audio")
                audio = audio / audio.abs().max()
            
            logger.info(f"Processed audio shape: {audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")

    def process_audio_enhanced(
        self, 
        audio_path: str,
        target_language: str = 'fra',
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Enhanced audio processing with diagnostics and language-specific improvements.
        """
        try:
            # Basic processing first
            audio = self.process_audio(audio_path)
            
            # Apply language-specific enhancements
            logger.info(f"Applying spectral enhancement for {target_language}")
            audio = self.apply_spectral_enhancement(audio, target_language)
            
            # Generate diagnostics if requested
            if return_diagnostics:
                if self.diagnostics is None:
                    logger.warning("Diagnostics requested but AudioDiagnostics not available")
                    return audio, {}
                    
                analysis = self.diagnostics.analyze_translation(audio, target_language)
                return audio, analysis
            
            return audio
            
        except Exception as e:
            error_msg = f"Enhanced audio processing failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)