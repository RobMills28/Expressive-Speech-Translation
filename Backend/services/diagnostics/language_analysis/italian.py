"""
Italian-specific audio analysis functionality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ItalianAnalyzer:
    """Analyzes Italian-specific audio characteristics."""

    def analyze(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze Italian-specific characteristics of audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            return {
                'gemination': self._analyze_italian_gemination(audio),
                'vowel_quality': self._analyze_italian_vowels(audio),
                'consonant_features': self._analyze_italian_consonants(audio),
                'prosodic_features': {
                    'stress_timing': self._analyze_italian_stress(audio),
                    'intonation': self._analyze_italian_intonation(audio),
                    'rhythm': self._analyze_italian_rhythm(audio)
                }
            }
        except Exception as e:
            logger.error(f"Italian analysis failed: {str(e)}")
            return {}

    def _analyze_italian_gemination(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Italian consonant gemination."""
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

            # Focus on consonant frequency regions
            consonant_region = mag[:, mag.shape[1]//2:]
            
            # Detect potential geminate consonants
            energy_profile = torch.mean(consonant_region, dim=1)
            peaks = self._find_consonant_peaks(energy_profile)
            
            # Analyze duration and intensity patterns
            gemination_patterns = self._analyze_gemination_patterns(peaks, energy_profile)
            
            return {
                'strength': float(gemination_patterns['strength']),
                'consistency': float(gemination_patterns['consistency']),
                'duration_contrast': float(gemination_patterns['duration_contrast']),
                'quality_score': float(gemination_patterns['overall_quality'])
            }
        except Exception as e:
            logger.error(f"Italian gemination analysis failed: {str(e)}")
            return {}

    def _analyze_italian_vowels(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Italian vowel characteristics."""
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
            
            # Focus on vowel formant regions
            formant_region = mag[:, :mag.shape[1]//3]
            
            # Analyze vowel quality characteristics
            clarity = float(torch.mean(formant_region).item())
            stability = float(1.0 - torch.std(formant_region) / (torch.mean(formant_region) + 1e-8))
            
            # Analyze formant structure
            formant_peaks = self._find_formant_peaks(formant_region)
            formant_stability = self._analyze_formant_stability(formant_peaks)
            
            return {
                'clarity': clarity,
                'stability': stability,
                'formant_structure': {
                    'peaks': formant_peaks[:3],  # First three formants
                    'stability': float(formant_stability)
                },
                'overall_quality': float((clarity + stability + formant_stability) / 3)
            }
        except Exception as e:
            logger.error(f"Italian vowel analysis failed: {str(e)}")
            return {}

    def _analyze_italian_consonants(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Italian consonant characteristics."""
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
            
            # Separate analysis for different consonant types
            plosive_analysis = self._analyze_plosives(mag)
            fricative_analysis = self._analyze_fricatives(mag)
            liquid_analysis = self._analyze_liquids(mag)
            
            return {
                'plosives': plosive_analysis,
                'fricatives': fricative_analysis,
                'liquids': liquid_analysis,
                'overall_precision': float(
                    (plosive_analysis['quality'] + 
                     fricative_analysis['quality'] + 
                     liquid_analysis['quality']) / 3
                )
            }
        except Exception as e:
            logger.error(f"Italian consonant analysis failed: {str(e)}")
            return {}

    def _analyze_italian_stress(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze Italian stress patterns."""
        try:
            # Convert to frequency domain
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
            peaks = self._find_stress_peaks(energy_contour)
            
            # Analyze stress characteristics
            if len(peaks) < 2:
                return {
                    'regularity': 0.0,
                    'contrast': 0.0,
                    'pattern_quality': 0.0
                }
            
            # Calculate regularity of stress intervals
            intervals = np.diff(peaks)
            regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8)
            
            # Calculate stress contrast
            peak_values = energy_contour[peaks]
            valley_values = torch.min(energy_contour)
            contrast = (torch.mean(peak_values) - valley_values) / (torch.mean(peak_values) + 1e-8)
            
            return {
                'regularity': float(regularity),
                'contrast': float(contrast),
                'pattern_quality': float((regularity + contrast) / 2)
            }
        except Exception as e:
            logger.error(f"Italian stress analysis failed: {str(e)}")
            return {'regularity': 0.0, 'contrast': 0.0, 'pattern_quality': 0.0}

    def _analyze_italian_intonation(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze Italian intonation patterns."""
        try:
            # Extract pitch contour
            audio_np = audio.squeeze().numpy()
            f0_contour = self._extract_pitch_contour(audio_np)
            
            if len(f0_contour) == 0:
                return {
                    'melodic_range': 0.0,
                    'contour_smoothness': 0.0,
                    'pattern_quality': 0.0
                }
            
            # Calculate melodic range
            melodic_range = (np.max(f0_contour) - np.min(f0_contour)) / (np.mean(f0_contour) + 1e-8)
            
            # Calculate contour smoothness
            contour_diff = np.diff(f0_contour)
            smoothness = 1.0 - np.std(contour_diff) / (np.mean(np.abs(contour_diff)) + 1e-8)
            
            return {
                'melodic_range': float(melodic_range),
                'contour_smoothness': float(smoothness),
                'pattern_quality': float((melodic_range + smoothness) / 2)
            }
        except Exception as e:
            logger.error(f"Italian intonation analysis failed: {str(e)}")
            return {'melodic_range': 0.0, 'contour_smoothness': 0.0, 'pattern_quality': 0.0}

    def _analyze_italian_rhythm(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze Italian rhythmic patterns."""
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
            
            # Extract rhythmic features
            energy_contour = torch.mean(mag, dim=1)
            rhythm_score = self._calculate_rhythm_score(energy_contour)
            
            return {
                'regularity': float(rhythm_score['regularity']),
                'pattern_strength': float(rhythm_score['strength']),
                'overall_quality': float(rhythm_score['quality'])
            }
        except Exception as e:
            logger.error(f"Italian rhythm analysis failed: {str(e)}")
            return {'regularity': 0.0, 'pattern_strength': 0.0, 'overall_quality': 0.0}

    # Helper methods
    def _find_consonant_peaks(self, energy_profile: torch.Tensor) -> List[int]:
        """Find peaks in consonant energy profile."""
        try:
            peaks = []
            for i in range(1, len(energy_profile)-1):
                if (energy_profile[i] > energy_profile[i-1] and 
                    energy_profile[i] > energy_profile[i+1]):
                    peaks.append(i)
            return peaks
        except Exception as e:
            logger.error(f"Consonant peak finding failed: {str(e)}")
            return []

    def _analyze_gemination_patterns(self, peaks: List[int], energy_profile: torch.Tensor) -> Dict[str, float]:
        """Analyze gemination patterns from consonant peaks."""
        try:
            if len(peaks) < 2:
                return {
                    'strength': 0.0,
                    'consistency': 0.0,
                    'duration_contrast': 0.0,
                    'overall_quality': 0.0
                }
            
            # Calculate peak characteristics
            peak_values = energy_profile[peaks]
            peak_intervals = np.diff(peaks)
            
            strength = float(torch.mean(peak_values).item())
            consistency = float(1.0 - torch.std(peak_values) / (torch.mean(peak_values) + 1e-8))
            duration_contrast = 1.0 - float(np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8))
            
            return {
                'strength': strength,
                'consistency': consistency,
                'duration_contrast': duration_contrast,
                'overall_quality': (strength + consistency + duration_contrast) / 3
            }
        except Exception as e:
            logger.error(f"Gemination pattern analysis failed: {str(e)}")
            return {'strength': 0.0, 'consistency': 0.0, 'duration_contrast': 0.0, 'overall_quality': 0.0}

    def _find_formant_peaks(self, formant_region: torch.Tensor) -> List[float]:
        """Find formant peaks in frequency spectrum."""
        try:
            mean_spectrum = torch.mean(formant_region, dim=0)
            peaks = []
            for i in range(1, len(mean_spectrum)-1):
                if (mean_spectrum[i] > mean_spectrum[i-1] and 
                    mean_spectrum[i] > mean_spectrum[i+1]):
                    peaks.append(float(i * 16000 / 4096))  # Convert to frequency
            return peaks
        except Exception as e:
            logger.error(f"Formant peak finding failed: {str(e)}")
            return []

    def _analyze_formant_stability(self, peaks: List[float]) -> float:
        """Analyze stability of formant frequencies."""
        try:
            if len(peaks) < 2:
                return 0.0
            formant_ratios = np.diff(peaks)
            return float(1.0 - np.std(formant_ratios) / (np.mean(formant_ratios) + 1e-8))
        except Exception as e:
            logger.error(f"Formant stability analysis failed: {str(e)}")
            return 0.0

    def _calculate_rhythm_score(self, energy_contour: torch.Tensor) -> Dict[str, float]:
        """Calculate rhythmic characteristics."""
        try:
            # Find peaks in energy contour
            peaks = []
            for i in range(1, len(energy_contour)-1):
                if (energy_contour[i] > energy_contour[i-1] and 
                    energy_contour[i] > energy_contour[i+1]):
                    peaks.append(i)
                    
            if len(peaks) < 2:
                return {
                    'regularity': 0.0,
                    'strength': 0.0,
                    'quality': 0.0
                }
                
            # Calculate rhythm metrics
            intervals = np.diff(peaks)
            peak_values = energy_contour[peaks]
            
            regularity = 1.0 - float(np.std(intervals) / (np.mean(intervals) + 1e-8))
            strength = float(torch.mean(peak_values).item())
            
            return {
                'regularity': regularity,
                'strength': strength,
                'quality': (regularity + strength) / 2
            }
        except Exception as e:
            logger.error(f"Rhythm score calculation failed: {str(e)}")
            return {'regularity': 0.0, 'strength': 0.0, 'quality': 0.0}