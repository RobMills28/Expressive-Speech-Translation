"""
Portuguese-specific audio analysis functionality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class PortugueseAnalyzer:
    """Analyzes Portuguese-specific audio characteristics."""

    def analyze(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze Portuguese-specific characteristics of audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            vowel_analysis = self._analyze_portuguese_nasalization(audio)
            consonant_analysis = self._analyze_portuguese_consonants(audio)
            
            return {
                'vowel_analysis': {
                    'nasalization': vowel_analysis,
                    'reduced_vowels': self._analyze_vowel_reduction(audio),
                    'diphthongs': self._analyze_portuguese_diphthongs(audio)
                },
                'consonant_features': {
                    'palatalization': self._analyze_palatalization(audio),
                    'sibilants': self._analyze_portuguese_sibilants(audio),
                    'rhotics': self._analyze_portuguese_r(audio)
                },
                'stress_patterns': self._analyze_portuguese_stress(audio),
                'intonation': self._analyze_portuguese_intonation(audio)
            }
        except Exception as e:
            logger.error(f"Portuguese analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_nasalization(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese nasal vowel characteristics."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Focus on nasal resonance frequencies (200-2500 Hz)
            nasal_region = mag[:, 10:125]
            
            # Analyze nasal characteristics
            nasal_strength = float(torch.mean(nasal_region).item())
            nasal_stability = float(1.0 - torch.std(nasal_region) / (torch.mean(nasal_region) + 1e-8))
            
            # Analyze formant structure specific to Portuguese nasals
            formant_structure = self._analyze_nasal_formants(nasal_region)
            
            return {
                'strength': nasal_strength,
                'stability': nasal_stability,
                'formant_structure': formant_structure,
                'quality': (nasal_strength + nasal_stability + formant_structure['quality']) / 3
            }
        except Exception as e:
            logger.error(f"Portuguese nasalization analysis failed: {str(e)}")
            return {}

    def _analyze_vowel_reduction(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese vowel reduction patterns."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Focus on vowel frequencies
            vowel_region = mag[:, :mag.shape[1]//3]
            
            # Calculate energy profile
            energy_profile = torch.mean(vowel_region, dim=1)
            
            # Find reduced vowel segments
            segments = self._find_vowel_segments(energy_profile)
            reduction_patterns = self._analyze_reduction_patterns(segments, energy_profile)
            
            return {
                'strength': float(reduction_patterns['strength']),
                'consistency': float(reduction_patterns['consistency']),
                'pattern_quality': float(reduction_patterns['quality'])
            }
        except Exception as e:
            logger.error(f"Vowel reduction analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_diphthongs(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese diphthong characteristics."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Find formant transitions
            formant_transitions = self._analyze_formant_transitions(mag)
            
            # Analyze diphthong characteristics
            smoothness = self._calculate_transition_smoothness(formant_transitions)
            duration = self._calculate_transition_duration(formant_transitions)
            
            return {
                'transition_smoothness': float(smoothness),
                'duration': float(duration),
                'quality': float((smoothness + duration) / 2)
            }
        except Exception as e:
            logger.error(f"Portuguese diphthong analysis failed: {str(e)}")
            return {}

    def _analyze_palatalization(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze palatalization in Portuguese consonants."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Focus on palatalization frequencies (2000-4000 Hz)
            palatal_region = mag[:, 100:200]
            
            # Analyze palatalization characteristics
            strength = float(torch.mean(palatal_region).item())
            consistency = float(1.0 - torch.std(palatal_region) / (torch.mean(palatal_region) + 1e-8))
            
            return {
                'strength': strength,
                'consistency': consistency,
                'quality': (strength + consistency) / 2
            }
        except Exception as e:
            logger.error(f"Palatalization analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_sibilants(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese sibilant characteristics."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Focus on sibilant frequencies (4000-8000 Hz)
            sibilant_region = mag[:, 200:400]
            
            # Calculate sibilant characteristics
            energy = float(torch.mean(sibilant_region).item())
            consistency = float(1.0 - torch.std(sibilant_region) / (torch.mean(sibilant_region) + 1e-8))
            
            return {
                'energy': energy,
                'consistency': consistency,
                'quality': (energy + consistency) / 2
            }
        except Exception as e:
            logger.error(f"Portuguese sibilant analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_r(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese rhotic sounds."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Analyze different rhotic variants
            trill_analysis = self._analyze_trill_characteristics(mag)
            tap_analysis = self._analyze_tap_characteristics(mag)
            
            return {
                'trill_quality': float(trill_analysis['quality']),
                'tap_quality': float(tap_analysis['quality']),
                'overall_quality': float((trill_analysis['quality'] + tap_analysis['quality']) / 2)
            }
        except Exception as e:
            logger.error(f"Portuguese rhotic analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_stress(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese stress patterns."""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Extract energy contour
            energy_contour = torch.mean(mag, dim=1)
            
            # Find stress peaks
            stress_peaks = self._find_stress_peaks(energy_contour)
            
            return {
                'regularity': float(self._calculate_stress_regularity(stress_peaks)),
                'contrast': float(self._calculate_stress_contrast(energy_contour, stress_peaks)),
                'pattern_quality': float(self._assess_stress_pattern(stress_peaks, energy_contour))
            }
        except Exception as e:
            logger.error(f"Portuguese stress analysis failed: {str(e)}")
            return {}

    def _analyze_portuguese_intonation(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese intonation patterns."""
        try:
            # Extract pitch contour
            pitch_contour = self._extract_pitch_contour(audio)
            
            # Analyze intonation characteristics
            contour_shape = self._analyze_contour_shape(pitch_contour)
            range_analysis = self._analyze_pitch_range(pitch_contour)
            
            return {
                'contour': contour_shape,
                'range': range_analysis,
                'quality': float((contour_shape['quality'] + range_analysis['quality']) / 2)
            }
        except Exception as e:
            logger.error(f"Portuguese intonation analysis failed: {str(e)}")
            return {}

    # Helper methods...
    def _analyze_nasal_formants(self, nasal_region: torch.Tensor) -> Dict[str, Any]:
        """Analyze formant structure of nasal vowels."""
        try:
            formant_peaks = []
            mean_spectrum = torch.mean(nasal_region, dim=0)
            
            # Find peaks in spectrum
            for i in range(1, len(mean_spectrum)-1):
                if (mean_spectrum[i] > mean_spectrum[i-1] and 
                    mean_spectrum[i] > mean_spectrum[i+1]):
                    formant_peaks.append((i, float(mean_spectrum[i].item())))
            
            if not formant_peaks:
                return {'quality': 0.0}
            
            # Calculate formant characteristics
            peak_frequencies = [p[0] * 16000 / 4096 for p in formant_peaks]  # Convert to Hz
            peak_amplitudes = [p[1] for p in formant_peaks]
            
            # Calculate quality metrics
            spacing_regularity = self._calculate_formant_spacing(peak_frequencies)
            amplitude_balance = self._calculate_amplitude_balance(peak_amplitudes)
            
            return {
                'frequencies': peak_frequencies[:3],  # First three formants
                'spacing_regularity': float(spacing_regularity),
                'amplitude_balance': float(amplitude_balance),
                'quality': float((spacing_regularity + amplitude_balance) / 2)
            }
        except Exception as e:
            logger.error(f"Nasal formant analysis failed: {str(e)}")
            return {'quality': 0.0}

    def _calculate_formant_spacing(self, frequencies: List[float]) -> float:
        """Calculate regularity of formant spacing."""
        try:
            if len(frequencies) < 2:
                return 0.0
            spacing = np.diff(frequencies)
            return float(1.0 - np.std(spacing) / (np.mean(spacing) + 1e-8))
        except Exception as e:
            logger.error(f"Formant spacing calculation failed: {str(e)}")
            return 0.0

    def _calculate_amplitude_balance(self, amplitudes: List[float]) -> float:
        """Calculate balance of formant amplitudes."""
        try:
            if not amplitudes:
                return 0.0
            return float(1.0 - np.std(amplitudes) / (np.mean(amplitudes) + 1e-8))
        except Exception as e:
            logger.error(f"Amplitude balance calculation failed: {str(e)}")
            return 0.0