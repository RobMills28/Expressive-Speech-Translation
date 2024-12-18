"""
German-specific audio analysis functionality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class GermanAnalyzer:
    """Analyzes German-specific audio characteristics."""

    def analyze(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze German-specific characteristics of audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            vowel_analysis = self._analyze_german_vowel_length(audio)
            consonant_analysis = self._analyze_german_consonants(audio)
            
            return {
                'vowel_analysis': vowel_analysis,
                'consonant_features': consonant_analysis,
                'word_stress': self._analyze_german_stress(audio),
                'glottal_stops': self._analyze_glottal_stops(audio),
                'final_devoicing': self._analyze_final_devoicing(audio),
                'schwa_realization': self._analyze_schwa(audio)
            }
        except Exception as e:
            logger.error(f"German analysis failed: {str(e)}")
            return {}

    def _analyze_german_vowel_length(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze German vowel length distinctions."""
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
        
            # Analyze formant regions for German vowels (approx. 300-2500 Hz)
            formant_region = mag[:, 15:125]
            
            # Analyze vowel durations
            energy_contour = torch.mean(formant_region, dim=1)
            segments = self._find_vowel_segments(energy_contour)
            
            # Calculate length distinctions
            length_ratios = self._calculate_length_ratios(segments)
            
            return {
                'formant_structure': {
                    'accuracy': float(torch.mean(formant_region).item()),
                    'stability': float(torch.std(formant_region).item())
                },
                'length_distinction': {
                    'ratio': float(length_ratios.mean()) if len(length_ratios) > 0 else 0.0,
                    'consistency': float(length_ratios.std()) if len(length_ratios) > 0 else 0.0
                },
                'quality': self._assess_vowel_quality(formant_region)
            }
            
        except Exception as e:
            logger.error(f"German vowel length analysis failed: {str(e)}")
            return {}

    def _analyze_german_consonants(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze German consonant clusters and characteristics."""
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
            
            # Analyze plosives
            plosive_score = self._analyze_plosives(consonant_region)
            
            # Analyze fricatives
            fricative_score = self._analyze_fricatives(consonant_region)
            
            # Analyze affricates
            affricate_score = self._analyze_affricates(consonant_region)
            
            return {
                'plosives': plosive_score,
                'fricatives': fricative_score,
                'affricates': affricate_score,
                'overall_precision': (plosive_score + fricative_score + affricate_score) / 3
            }
            
        except Exception as e:
            logger.error(f"German consonant analysis failed: {str(e)}")
            return {}

    def _analyze_german_stress(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze German word stress patterns."""
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
            
            # Analyze energy distribution for stress patterns
            energy_contour = torch.mean(mag, dim=1)
            stress_variation = torch.std(energy_contour) / torch.mean(energy_contour)
            
            # Calculate rhythmic regularity
            rhythm_score = self._calculate_rhythm_score(energy_contour)
            
            return {
                'stress_contrast': float(min(1.0, stress_variation)),
                'rhythm_regularity': float(rhythm_score),
                'overall_stress_quality': float((min(1.0, stress_variation) + rhythm_score) / 2)
            }
            
        except Exception as e:
            logger.error(f"German stress analysis failed: {str(e)}")
            return {'stress_contrast': 0.0, 'rhythm_regularity': 0.0, 'overall_stress_quality': 0.0}

    def _analyze_glottal_stops(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze German glottal stops."""
        try:
            # Find sudden amplitude drops followed by sharp rises
            audio_np = audio.squeeze().numpy()
            envelope = np.abs(audio_np)
            
            # Detect potential glottal stops
            diff = np.diff(envelope)
            potential_stops = np.where((diff[:-1] < -np.std(diff)) & (diff[1:] > np.std(diff)))[0]
            
            if len(potential_stops) == 0:
                return {'presence': 0.0, 'clarity': 0.0}
            
            # Analyze stop characteristics
            stop_strengths = []
            for stop_idx in potential_stops:
                if stop_idx > 10 and stop_idx < len(envelope) - 10:
                    pre_stop = np.mean(envelope[stop_idx-10:stop_idx])
                    post_stop = np.mean(envelope[stop_idx+1:stop_idx+11])
                    strength = abs(post_stop - pre_stop) / max(pre_stop, post_stop)
                    stop_strengths.append(strength)
            
            if not stop_strengths:
                return {'presence': 0.0, 'clarity': 0.0}
                
            return {
                'presence': float(len(stop_strengths) / len(audio_np) * 1000),
                'clarity': float(np.mean(stop_strengths))
            }
            
        except Exception as e:
            logger.error(f"Glottal stop analysis failed: {str(e)}")
            return {'presence': 0.0, 'clarity': 0.0}

    def _analyze_final_devoicing(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze German final devoicing."""
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
            
            # Analyze end segments for voicing
            segment_length = 20  # frames
            end_segments = mag[:, -segment_length:]
            
            # Calculate voicing characteristics
            low_freq_energy = torch.mean(end_segments[:, :10])
            high_freq_energy = torch.mean(end_segments[:, 10:])
            
            voicing_ratio = low_freq_energy / (high_freq_energy + 1e-8)
            
            return {
                'devoicing_strength': float(1.0 - min(1.0, voicing_ratio)),
                'consistency': float(1.0 - torch.std(end_segments) / (torch.mean(end_segments) + 1e-8))
            }
            
        except Exception as e:
            logger.error(f"Final devoicing analysis failed: {str(e)}")
            return {'devoicing_strength': 0.0, 'consistency': 0.0}

    def _analyze_schwa(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze German schwa realization."""
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
            
            # Focus on schwa frequency range (approximately 1200-1600 Hz)
            schwa_region = mag[:, 60:80]
            
            # Calculate schwa characteristics
            strength = float(torch.mean(schwa_region).item())
            stability = float(1.0 - torch.std(schwa_region) / (torch.mean(schwa_region) + 1e-8))
            
            return {
                'presence': strength,
                'stability': stability,
                'quality': (strength + stability) / 2
            }
            
        except Exception as e:
            logger.error(f"Schwa analysis failed: {str(e)}")
            return {'presence': 0.0, 'stability': 0.0, 'quality': 0.0}

    # Helper methods
    def _find_vowel_segments(self, energy_contour: torch.Tensor) -> List[Tuple[int, int]]:
        """Find vowel segments in energy contour."""
        try:
            threshold = torch.mean(energy_contour) * 0.6
            is_vowel = energy_contour > threshold
            
            segments = []
            start = None
            
            for i, val in enumerate(is_vowel):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    segments.append((start, i))
                    start = None
            
            if start is not None:
                segments.append((start, len(is_vowel)))
            
            return segments
        except Exception as e:
            logger.error(f"Vowel segment finding failed: {str(e)}")
            return []

    def _calculate_length_ratios(self, segments: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate vowel length ratios."""
        try:
            lengths = np.array([end - start for start, end in segments])
            return lengths[1:] / lengths[:-1] if len(lengths) > 1 else np.array([])
        except Exception as e:
            logger.error(f"Length ratio calculation failed: {str(e)}")
            return np.array([])

    def _calculate_rhythm_score(self, energy_contour: torch.Tensor) -> float:
        """Calculate rhythmic regularity score."""
        try:
            # Find peaks in energy contour
            peaks = []
            for i in range(1, len(energy_contour)-1):
                if energy_contour[i] > energy_contour[i-1] and energy_contour[i] > energy_contour[i+1]:
                    peaks.append(i)
            
            if len(peaks) < 2:
                return 0.0
            
            # Calculate inter-peak intervals
            intervals = np.diff(peaks)
            
            # Calculate regularity as inverse of interval variance
            regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8)
            return float(max(0.0, min(1.0, regularity)))
            
        except Exception as e:
            logger.error(f"Rhythm score calculation failed: {str(e)}")
            return 0.0

    def _assess_vowel_quality(self, formant_region: torch.Tensor) -> Dict[str, float]:
        """Assess vowel quality metrics."""
        try:
            return {
                'clarity': float(torch.mean(formant_region).item()),
                'stability': float(1.0 - torch.std(formant_region) / (torch.mean(formant_region) + 1e-8)),
                'formant_definition': float(torch.max(formant_region).item() / (torch.mean(formant_region) + 1e-8))
            }
        except Exception as e:
            logger.error(f"Vowel quality assessment failed: {str(e)}")
            return {'clarity': 0.0, 'stability': 0.0, 'formant_definition': 0.0}