"""
Spanish-specific audio analysis functionality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class SpanishAnalyzer:
    """Analyzes Spanish-specific audio characteristics."""

    def analyze(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze Spanish-specific characteristics of audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            return {
                'phoneme_analysis': {
                    'trilled_r': self._analyze_spanish_trill(audio),
                    'interdental_theta': self._analyze_interdental(audio),
                    'stop_consonants': self._analyze_spanish_stops(audio)
                },
                'syllable_timing': self._analyze_spanish_timing(audio),
                'intonation_patterns': self._analyze_spanish_intonation(audio),
                'vowel_clarity': self._analyze_spanish_vowels(audio),
                'stress_patterns': self._analyze_spanish_stress(audio)
            }
        except Exception as e:
            logger.error(f"Spanish analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_trill(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish trilled R characteristics."""
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
            
            # Calculate trill characteristics from energy modulation
            trill_features = self._analyze_trill_features(mag)
            
            # Analyze strength and regularity of trills
            trill_quality = self._assess_trill_quality(trill_features)
            
            # Combine results
            return {
                'features': trill_features,
                'strength': float(trill_quality['strength']),
                'regularity': float(trill_quality['regularity']),
                'duration': float(trill_quality['duration']),
                'overall_quality': float(trill_quality['overall_quality'])
            }
        except Exception as e:
            logger.error(f"Spanish trill analysis failed: {str(e)}")
            return {}

    def _analyze_interdental(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish interdental consonant (θ) characteristics."""
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
            
            # Focus on high-frequency content characteristic of interdental sounds
            high_freq_region = mag[:, mag.shape[1]//2:]
            
            # Calculate interdental characteristics
            energy = float(torch.mean(high_freq_region).item())
            consistency = float(1.0 - torch.std(high_freq_region) / (torch.mean(high_freq_region) + 1e-8))
            
            # Analyze spectral shape for characteristic interdental pattern
            spectral_shape = self._analyze_interdental_spectrum(high_freq_region)
            
            return {
                'energy': energy,
                'consistency': consistency,
                'spectral_shape': spectral_shape,
                'quality': float((energy + consistency + spectral_shape['quality']) / 3)
            }
        except Exception as e:
            logger.error(f"Interdental analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_stops(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish stop consonant characteristics."""
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
            
            # Analyze different stop types
            voiceless_stops = self._analyze_voiceless_stops(mag)
            voiced_stops = self._analyze_voiced_stops(mag)
            
            return {
                'voiceless_stops': voiceless_stops,
                'voiced_stops': voiced_stops,
                'overall_quality': float((
                    voiceless_stops['quality'] + 
                    voiced_stops['quality']
                ) / 2)
            }
        except Exception as e:
            logger.error(f"Spanish stop analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_timing(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish syllable timing patterns."""
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
            
            # Find syllable boundaries
            syllable_boundaries = self._find_syllable_boundaries(energy_contour)
            
            # Calculate timing characteristics
            timing_metrics = self._analyze_timing_metrics(syllable_boundaries)
            
            return {
                'regularity': float(timing_metrics['regularity']),
                'rate': float(timing_metrics['rate']),
                'pattern_quality': float(timing_metrics['quality'])
            }
        except Exception as e:
            logger.error(f"Spanish timing analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_intonation(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish intonation patterns."""
        try:
            # Extract pitch contour
            pitch_contour = self._extract_pitch_contour(audio)
            
            # Analyze intonation patterns
            contour_analysis = self._analyze_pitch_patterns(pitch_contour)
            
            # Analyze melodic features
            melodic_features = self._analyze_melodic_features(pitch_contour)
            
            return {
                'contour_patterns': contour_analysis,
                'melodic_features': melodic_features,
                'overall_quality': float((
                    contour_analysis['quality'] + 
                    melodic_features['quality']
                ) / 2)
            }
        except Exception as e:
            logger.error(f"Spanish intonation analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_vowels(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish vowel clarity and stability."""
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
            
            # Calculate vowel characteristics
            clarity = float(torch.mean(vowel_region).item())
            stability = float(1.0 - torch.std(vowel_region) / (torch.mean(vowel_region) + 1e-8))
            
            # Analyze formant structure
            formant_analysis = self._analyze_vowel_formants(vowel_region)
            
            return {
                'clarity': clarity,
                'stability': stability,
                'formant_structure': formant_analysis,
                'quality': float((clarity + stability + formant_analysis['quality']) / 3)
            }
        except Exception as e:
            logger.error(f"Spanish vowel analysis failed: {str(e)}")
            return {}

    def _analyze_spanish_stress(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish stress patterns."""
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
            
            # Calculate stress pattern characteristics
            stress_patterns = self._analyze_stress_patterns(stress_peaks, energy_contour)
            
            return {
                'regularity': float(stress_patterns['regularity']),
                'contrast': float(stress_patterns['contrast']),
                'pattern_quality': float(stress_patterns['quality'])
            }
        except Exception as e:
            logger.error(f"Spanish stress analysis failed: {str(e)}")
            return {}

    # Helper methods
    def _analyze_trill_features(self, mag: torch.Tensor) -> Dict[str, Any]:
        """Analyze trill features from spectrogram."""
        try:
            # Focus on trill frequency region (20-40 Hz modulation)
            trill_energy = torch.mean(mag[:, mag.shape[1]//4:mag.shape[1]//2], dim=1)
            
            # Find modulation peaks
            peaks = self._find_modulation_peaks(trill_energy)
            
            if not peaks:
                return {'modulation_rate': 0.0, 'strength': 0.0}
            
            # Calculate modulation rate
            modulation_rate = len(peaks) / (len(trill_energy) / 16000)  # Convert to Hz
            
            # Calculate modulation strength
            strength = np.mean([trill_energy[p] for p in peaks])
            
            return {
                'modulation_rate': float(modulation_rate),
                'strength': float(strength)
            }
        except Exception as e:
            logger.error(f"Trill feature analysis failed: {str(e)}")
            return {'modulation_rate': 0.0, 'strength': 0.0}

    def _assess_trill_quality(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of trill features."""
        try:
            # Ideal Spanish trill rate is around 25-30 Hz
            rate_quality = 1.0 - abs(features['modulation_rate'] - 27.5) / 27.5
            
            # Combine with strength for overall quality
            return {
                'strength': float(min(1.0, features['strength'])),
                'regularity': float(max(0.0, rate_quality)),
                'duration': float(min(1.0, features['strength'] * rate_quality)),
                'overall_quality': float((min(1.0, features['strength']) + max(0.0, rate_quality)) / 2)
            }
        except Exception as e:
            logger.error(f"Trill quality assessment failed: {str(e)}")
            return {'strength': 0.0, 'regularity': 0.0, 'duration': 0.0, 'overall_quality': 0.0}

    def _find_modulation_peaks(self, energy: torch.Tensor) -> List[int]:
        """Find modulation peaks in energy contour."""
        try:
            peaks = []
            for i in range(1, len(energy)-1):
                if energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                    peaks.append(i)
            return peaks
        except Exception as e:
            logger.error(f"Modulation peak finding failed: {str(e)}")
            return []

    def _extract_pitch_contour(self, audio: torch.Tensor) -> np.ndarray:
        """Extract pitch contour using autocorrelation."""
        try:
            audio_np = audio.squeeze().numpy()
            frame_length = 1024
            hop_length = 512
            
            # Calculate pitch for each frame
            num_frames = (len(audio_np) - frame_length) // hop_length + 1
            pitch_contour = np.zeros(num_frames)
            
            for i in range(num_frames):
                frame = audio_np[i*hop_length:i*hop_length+frame_length]
                # Use autocorrelation for pitch detection
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]
                
                # Find first peak after initial drop
                peak_idx = np.argmax(corr[50:]) + 50
                if corr[peak_idx] > 0.1:  # Threshold for voiced speech
                    pitch_contour[i] = 16000 / peak_idx  # Convert to Hz
                
            return pitch_contour
        except Exception as e:
            logger.error(f"Pitch contour extraction failed: {str(e)}")
            return np.array([])

    def _analyze_pitch_patterns(self, pitch_contour: np.ndarray) -> Dict[str, Any]:
        """Analyze pitch patterns in contour."""
        try:
            if len(pitch_contour) == 0:
                return {'quality': 0.0}
            
            # Calculate contour characteristics
            mean_pitch = np.mean(pitch_contour[pitch_contour > 0])
            std_pitch = np.std(pitch_contour[pitch_contour > 0])
            
            # Analyze pitch movement patterns
            rises = np.sum(np.diff(pitch_contour) > 0)
            falls = np.sum(np.diff(pitch_contour) < 0)
            
            return {
                'mean_pitch': float(mean_pitch),
                'pitch_variability': float(std_pitch / mean_pitch),
                'rise_fall_ratio': float(rises / (falls + 1e-8)),
                'quality': float(1.0 - std_pitch / (mean_pitch + 1e-8))
            }
        except Exception as e:
            logger.error(f"Pitch pattern analysis failed: {str(e)}")
            return {'quality': 0.0}

    def _analyze_stress_patterns(self, peaks: List[int], energy_contour: torch.Tensor) -> Dict[str, float]:
        """Analyze stress patterns from energy peaks."""
        try:
            if not peaks or len(peaks) < 2:
                return {'regularity': 0.0, 'contrast': 0.0, 'quality': 0.0}
            
            # Calculate intervals between stress peaks
            intervals = np.diff(peaks)
            regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8)
            
            # Calculate stress contrast
            peak_energies = energy_contour[peaks]
            valley_energies = torch.min(energy_contour)
            contrast = (torch.mean(peak_energies) - valley_energies) / (torch.mean(peak_energies) + 1e-8)
            
            quality = (float(regularity) + float(contrast)) / 2
            
            return {
                'regularity': float(regularity),
                'contrast': float(contrast),
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Stress pattern analysis failed: {str(e)}")
            return {'regularity': 0.0, 'contrast': 0.0, 'quality': 0.0}

    def _analyze_interdental_spectrum(self, high_freq_region: torch.Tensor) -> Dict[str, float]:
        """Analyze spectral characteristics of interdental sounds."""
        try:
            # Calculate spectral centroid
            freqs = torch.linspace(0, 1, high_freq_region.shape[1])
            spec_centroid = torch.sum(high_freq_region * freqs, dim=1) / (torch.sum(high_freq_region, dim=1) + 1e-8)
            
            # Calculate spectral spread
            spread = torch.sqrt(
                torch.sum(high_freq_region * (freqs - spec_centroid.unsqueeze(1))**2, dim=1) /
                (torch.sum(high_freq_region, dim=1) + 1e-8)
            )
            
            # Evaluate quality based on typical interdental characteristics
            centroid_quality = 1.0 - torch.abs(torch.mean(spec_centroid) - 0.7).item()  # Expected centroid around 0.7
            spread_quality = 1.0 - torch.mean(spread).item()  # Prefer concentrated energy
            
            quality = (centroid_quality + spread_quality) / 2
            
            return {
                'centroid': float(torch.mean(spec_centroid).item()),
                'spread': float(torch.mean(spread).item()),
                'quality': float(quality)
            }
        except Exception as e:
            logger.error(f"Interdental spectrum analysis failed: {str(e)}")
            return {'centroid': 0.0, 'spread': 0.0, 'quality': 0.0}

    def _analyze_voiceless_stops(self, mag: torch.Tensor) -> Dict[str, float]:
        """Analyze voiceless stop consonants (p, t, k)."""
        try:
            # Look for sudden energy bursts followed by aspiration
            energy_profile = torch.mean(mag, dim=1)
            bursts = self._find_burst_points(energy_profile)
            
            if not bursts:
                return {'strength': 0.0, 'precision': 0.0, 'quality': 0.0}
            
            # Analyze burst characteristics
            burst_strength = torch.mean(energy_profile[bursts]).item()
            burst_precision = 1.0 - torch.std(energy_profile[bursts]).item() / (burst_strength + 1e-8)
            
            return {
                'strength': float(min(1.0, burst_strength)),
                'precision': float(burst_precision),
                'quality': float((min(1.0, burst_strength) + burst_precision) / 2)
            }
        except Exception as e:
            logger.error(f"Voiceless stop analysis failed: {str(e)}")
            return {'strength': 0.0, 'precision': 0.0, 'quality': 0.0}

    def _analyze_voiced_stops(self, mag: torch.Tensor) -> Dict[str, float]:
        """Analyze voiced stop consonants (b, d, g)."""
        try:
            # Look for voicing during closure and weaker bursts
            energy_profile = torch.mean(mag[:, :mag.shape[1]//4], dim=1)  # Low frequency for voicing
            closure_regions = self._find_closure_regions(energy_profile)
            
            if not closure_regions:
                return {'voicing': 0.0, 'precision': 0.0, 'quality': 0.0}
            
            # Analyze closure and voicing characteristics
            voicing_strength = torch.mean(energy_profile[closure_regions]).item()
            precision = 1.0 - torch.std(energy_profile[closure_regions]).item() / (voicing_strength + 1e-8)
            
            return {
                'voicing': float(min(1.0, voicing_strength)),
                'precision': float(precision),
                'quality': float((min(1.0, voicing_strength) + precision) / 2)
            }
        except Exception as e:
            logger.error(f"Voiced stop analysis failed: {str(e)}")
            return {'voicing': 0.0, 'precision': 0.0, 'quality': 0.0}

    def _find_burst_points(self, energy_profile: torch.Tensor) -> List[int]:
        """Find burst points in energy profile."""
        try:
            bursts = []
            threshold = torch.mean(energy_profile) + torch.std(energy_profile)
            
            for i in range(1, len(energy_profile)-1):
                if (energy_profile[i] > threshold and
                    energy_profile[i] > energy_profile[i-1] and
                    energy_profile[i] > energy_profile[i+1]):
                    bursts.append(i)
            
            return bursts
        except Exception as e:
            logger.error(f"Burst point finding failed: {str(e)}")
            return []

    def _find_closure_regions(self, energy_profile: torch.Tensor) -> List[int]:
        """Find closure regions in energy profile."""
        try:
            closure_points = []
            threshold = torch.mean(energy_profile) * 0.5
            
            for i in range(len(energy_profile)):
                if energy_profile[i] < threshold:
                    closure_points.append(i)
            
            return closure_points
        except Exception as e:
            logger.error(f"Closure region finding failed: {str(e)}")
            return []

    def _analyze_melodic_features(self, pitch_contour: np.ndarray) -> Dict[str, float]:
        """Analyze melodic features of intonation."""
        try:
            if len(pitch_contour) == 0:
                return {'range': 0.0, 'movement': 0.0, 'quality': 0.0}
            
            # Calculate pitch range
            valid_pitch = pitch_contour[pitch_contour > 0]
            if len(valid_pitch) == 0:
                return {'range': 0.0, 'movement': 0.0, 'quality': 0.0}
                
            pitch_range = (np.max(valid_pitch) - np.min(valid_pitch)) / np.mean(valid_pitch)
            
            # Calculate pitch movement
            pitch_movement = np.sum(np.abs(np.diff(valid_pitch))) / len(valid_pitch)
            
            # Normalize and combine scores
            range_score = min(1.0, pitch_range)
            movement_score = min(1.0, pitch_movement / 50)  # Normalize to reasonable range
            
            return {
                'range': float(range_score),
                'movement': float(movement_score),
                'quality': float((range_score + movement_score) / 2)
            }
        except Exception as e:
            logger.error(f"Melodic feature analysis failed: {str(e)}")
            return {'range': 0.0, 'movement': 0.0, 'quality': 0.0}