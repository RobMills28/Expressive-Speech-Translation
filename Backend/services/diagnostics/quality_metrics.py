"""
Quality metrics and scoring functionality for audio analysis.
"""
from enum import Enum
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
import logging


logger = logging.getLogger(__name__)

class AudioQualityLevel(Enum):
    """Define quality levels for audio assessment"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class FrequencyBand:
    """Frequency band information"""
    name: str
    low_freq: float
    high_freq: float
    description: str
    perceptual_features: List[str]

class QualityMetrics:
    """Audio quality metrics calculation"""

    @staticmethod
    def calculate_robotic_score(audio: torch.Tensor) -> float:
        """Calculate roboticness score (1-5 scale) based on harmonic regularity"""
        try:
            # Convert to frequency domain
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
        
            # Calculate harmonic regularity
            harmonic_peaks = torch.max(mag, dim=1)[0]
            peak_variance = torch.var(harmonic_peaks)
        
            # More robotic speech has very regular harmonics (lower variance)
            score = 5.0 - (4.0 * (1.0 - peak_variance / torch.mean(harmonic_peaks)))
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Robotic score calculation failed: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_pronunciation_score(audio: torch.Tensor) -> float:
        """Calculate pronunciation score based on spectral clarity and formant strength"""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
        
            # Analyze formant regions (approximately 500-3500 Hz)
            formant_region = mag[:, 50:350]  # Adjust indices based on sampling rate
            formant_strength = torch.mean(formant_region)
        
            # Normalize to 1-5 scale
            score = 1.0 + 4.0 * (formant_strength / torch.max(mag))
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Pronunciation score calculation failed: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_clarity_score(audio: torch.Tensor) -> float:
        """Calculate clarity score based on signal-to-noise ratio and spectral contrast"""
        try:
            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(audio ** 2))
        
            # Calculate spectral contrast
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            contrast = torch.mean(torch.max(mag, dim=1)[0] / torch.mean(mag, dim=1))
        
            # Combine metrics
            score = 1.0 + 2.0 * float(rms) + 2.0 * float(contrast / 10.0)
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Clarity score calculation failed: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_noise_score(audio: torch.Tensor) -> float:
        """Calculate noise score, distinguishing breathiness from noise"""
        try:
            # Calculate signal power
            signal_power = torch.mean(audio ** 2)
        
            # Separate voice characteristics from noise
            sorted_amplitudes = torch.sort(torch.abs(audio.squeeze()))[0]
        
            # Use first 5% for noise floor instead of 10%
            noise_floor = torch.mean(sorted_amplitudes[:int(len(sorted_amplitudes)*0.05)]) ** 2
        
            # Upper spectral envelope for breathiness
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            high_freq_content = torch.mean(mag[:, mag.shape[1]//2:])  # Upper half of spectrum
        
            # Calculate SNR excluding breathy components
            if noise_floor > 0:
                snr = 10 * torch.log10(signal_power / noise_floor)
                # Adjust score based on both SNR and breathiness
                score = 1.0 + 4.0 * (torch.clamp(snr, 0, 30) / 30) * (1 - high_freq_content/torch.max(mag))
            else:
                score = 5.0
        
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Noise score calculation failed: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_consistency_score(audio: torch.Tensor) -> float:
        """Calculate voice consistency score based on amplitude and spectral stability"""
        try:
            # Analyze amplitude stability
            segments = torch.split(audio.squeeze(), 2048)
            segment_rms = torch.tensor([torch.sqrt(torch.mean(seg ** 2)) for seg in segments if len(seg) == 2048])
            amplitude_stability = 1.0 - torch.std(segment_rms) / torch.mean(segment_rms)
        
            # Analyze spectral stability
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            spectral_stability = 1.0 - torch.std(torch.mean(mag, dim=1)) / torch.mean(mag)
        
            # Combine metrics
            score = 1.0 + 4.0 * (amplitude_stability + spectral_stability) / 2
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Consistency score calculation failed: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_balance_score(spectral_bands: Dict[str, float]) -> float:
        """Calculate spectral balance score based on energy distribution"""
        try:
            if not spectral_bands or not all(k in spectral_bands for k in ['low', 'mid', 'high']):
                return 0.0
            
            # Calculate relative energies
            total_energy = sum(spectral_bands.values())
            if total_energy == 0:
                return 0.0
            
            ratios = {k: v/total_energy for k, v in spectral_bands.items()}
        
            # Ideal ratios for speech
            ideal_ratios = {'low': 0.3, 'mid': 0.5, 'high': 0.2}
        
            # Calculate deviation from ideal
            deviation = sum(abs(ratios[k] - ideal_ratios[k]) for k in ratios)
        
            # Convert to 1-5 score (lower deviation is better)
            score = 5.0 * (1.0 - deviation)
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Balance score calculation failed: {str(e)}")
            return 0.0