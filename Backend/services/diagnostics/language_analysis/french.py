"""
French-specific audio analysis functionality.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FrenchAnalyzer:
    """Analyzes French-specific audio characteristics."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 4096
        self.hop_length = 1024
        self.win_length = 4096

    def analyze(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze French-specific characteristics of audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, Any]: Analysis results including nasalization, liaison, prosody, 
                          and vowel quality metrics
        """
        try:
            # Basic validation
            if audio is None or audio.numel() == 0:
                raise ValueError("Empty audio tensor provided")
            
            # Ensure audio is 2D (batch_size, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() > 2:
                audio = audio.squeeze()
                if audio.dim() > 2:
                    raise ValueError("Audio tensor has too many dimensions")

            # Get spectral representation for analysis
            spec = self._get_spectrogram(audio)
            
            # Perform comprehensive analysis
            nasalization = self._analyze_french_nasalization(spec)
            liaison = self._analyze_french_liaison(spec)
            prosody = self._analyze_french_prosody(spec)
            vowel_quality = self._analyze_french_vowels(spec)
            
            return {
                'nasalization': nasalization,
                'liaison': liaison,
                'prosody': prosody,
                'vowel_quality': vowel_quality
            }
        except Exception as e:
            logger.error(f"French analysis failed: {str(e)}")
            return {
                'nasalization': {
                    'nasal_resonance': {'strength': 0.0, 'stability': 0.0, 'peak_frequencies': []},
                    'quality_assessment': {'authenticity': 0.0, 'consistency': 0.0, 'distinction': 0.0},
                    'description': "Analysis failed"
                },
                'liaison': {'detected': False, 'confidence': 0.0, 'description': "Analysis failed"},
                'prosody': {'score': 0.0, 'rhythm_quality': 0.0, 'intonation_quality': 0.0},
                'vowel_quality': {'quality_score': 0.0, 'formant_structure': 0.0, 'description': "Analysis failed"}
            }

    def _get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate spectrogram for analysis."""
        try:
            window = torch.hann_window(self.win_length).to(audio.device)
            spec = torch.stft(
                audio.squeeze(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True
            )
            return torch.abs(spec)
        except Exception as e:
            logger.error(f"Spectrogram generation failed: {str(e)}")
            raise

    def _analyze_french_nasalization(self, spec: torch.Tensor) -> Dict[str, Any]:
        """Analyze French nasal vowel characteristics."""
        try:
            # Focus on nasal frequencies (500-2000 Hz)
            nasal_band = spec[:, 25:100]  # Approximate band for nasal resonances
            
            # Calculate resonance characteristics
            strength = float(torch.mean(nasal_band).item())
            stability = float(torch.std(nasal_band).item())
            peak_frequencies = self._find_peak_frequencies(nasal_band)
            
            # Quality assessments
            authenticity = self._assess_nasal_authenticity(nasal_band)
            consistency = self._assess_nasal_consistency(nasal_band)
            distinction = self._assess_nasal_distinction(spec)
            
            analysis = {
                'nasal_resonance': {
                    'strength': strength,
                    'stability': stability,
                    'peak_frequencies': peak_frequencies
                },
                'quality_assessment': {
                    'authenticity': authenticity,
                    'consistency': consistency,
                    'distinction': distinction
                }
            }
            
            # Add descriptive analysis
            analysis['description'] = self._describe_nasal_qualities(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"French nasalization analysis failed: {str(e)}")
            return {
                'nasal_resonance': {'strength': 0.0, 'stability': 0.0, 'peak_frequencies': []},
                'quality_assessment': {'authenticity': 0.0, 'consistency': 0.0, 'distinction': 0.0},
                'description': "Analysis failed"
            }

    def _find_peak_frequencies(self, band: torch.Tensor) -> List[float]:
        """Find peak frequencies in a spectral band."""
        try:
            peaks = []
            # Use local maxima to find peaks
            for i in range(1, band.shape[1]-1):
                if torch.all(band[:, i] > band[:, i-1]) and torch.all(band[:, i] > band[:, i+1]):
                    freq = i * self.sample_rate / self.n_fft
                    peaks.append(float(freq))
            return sorted(peaks)[:5]  # Return top 5 peaks
        except Exception as e:
            logger.error(f"Peak frequency analysis failed: {str(e)}")
            return []

    def _assess_nasal_authenticity(self, nasal_band: torch.Tensor) -> float:
        """Assess authenticity of nasal resonances."""
        try:
            # Calculate energy distribution in nasal band
            energy_profile = torch.mean(nasal_band, dim=0)
            
            # Create typical French nasal vowel profile
            typical_profile = torch.exp(-torch.linspace(0, 2, nasal_band.shape[1]))
            
            # Normalize profiles
            energy_profile_norm = (energy_profile - energy_profile.mean()) / (energy_profile.std() + 1e-6)
            typical_profile_norm = (typical_profile - typical_profile.mean()) / (typical_profile.std() + 1e-6)
            
            # Calculate correlation manually
            correlation = torch.sum(energy_profile_norm * typical_profile_norm) / (len(energy_profile_norm) - 1)
            
            return float(max(0.0, min(1.0, correlation.item())))
        except Exception as e:
            logger.error(f"Nasal authenticity assessment failed: {str(e)}")
            return 0.0

    def _assess_nasal_consistency(self, nasal_band: torch.Tensor) -> float:
        """Assess consistency of nasal resonances."""
        try:
            # Measure stability over time
            temporal_variance = torch.std(torch.mean(nasal_band, dim=1))
            mean_energy = torch.mean(nasal_band)
            
            # Calculate stability score
            stability = 1.0 - (temporal_variance / (mean_energy + 1e-8))
            return float(max(0.0, min(1.0, stability.item())))
        except Exception as e:
            logger.error(f"Nasal consistency assessment failed: {str(e)}")
            return 0.0

    def _assess_nasal_distinction(self, spec: torch.Tensor) -> float:
        """Assess distinction between nasal and non-nasal segments."""
        try:
            # Compare energy in nasal vs. oral frequency bands
            nasal_region = torch.mean(spec[:, 25:100])  # Nasal frequencies
            oral_region = torch.mean(spec[:, 100:200])  # Oral frequencies
            
            # Calculate distinction score
            distinction = torch.abs(nasal_region - oral_region) / (nasal_region + oral_region + 1e-8)
            return float(min(1.0, distinction.item()))
        except Exception as e:
            logger.error(f"Nasal distinction assessment failed: {str(e)}")
            return 0.0

    def _analyze_french_liaison(self, spec: torch.Tensor) -> Dict[str, Any]:
        """Analyze French liaison patterns."""
        try:
            # Segment the spectrogram for analysis
            segment_length = 20  # frames
            segments = torch.split(spec, segment_length, dim=0)
            
            if len(segments) < 2:
                return {'detected': False, 'confidence': 0.0, 'description': "Audio too short for liaison analysis"}
            
            # Calculate stability between segments
            stability_scores = []
            for i in range(len(segments)-1):
                if segments[i].shape == segments[i+1].shape:
                    seg1_flat = segments[i].flatten()
                    seg2_flat = segments[i+1].flatten()
                    
                    # Normalize segments
                    seg1_norm = (seg1_flat - seg1_flat.mean()) / (seg1_flat.std() + 1e-6)
                    seg2_norm = (seg2_flat - seg2_flat.mean()) / (seg2_flat.std() + 1e-6)
                    
                    # Calculate correlation manually
                    correlation = torch.sum(seg1_norm * seg2_norm) / (len(seg1_norm) - 1)
                    stability_scores.append(float(correlation))
            
            if not stability_scores:
                return {'detected': False, 'confidence': 0.0, 'description': "No valid segments for analysis"}
            
            # Calculate overall liaison metrics
            average_stability = np.mean(stability_scores)
            confidence = max(0.0, min(1.0, average_stability))
            detected = confidence > 0.7
            
            description = (
                "Clear liaison patterns detected" if detected else
                "Weak or inconsistent liaison patterns"
            )
            
            return {
                'detected': detected,
                'confidence': confidence,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"French liaison analysis failed: {str(e)}")
            return {'detected': False, 'confidence': 0.0, 'description': "Analysis failed"}

    def _analyze_french_prosody(self, spec: torch.Tensor) -> Dict[str, Any]:
        """Analyze French prosodic patterns."""
        try:
            # Calculate energy contour
            energy_contour = torch.mean(spec, dim=1)
            
            # Analyze rhythm
            temporal_variance = torch.std(energy_contour)
            mean_energy = torch.mean(energy_contour)
            rhythm_score = float(1.0 - (temporal_variance / (mean_energy + 1e-8)))
            
            # Analyze intonation patterns
            intonation_score = self._analyze_intonation_pattern(spec)
            
            # Calculate overall prosody score
            overall_score = (rhythm_score + intonation_score) / 2
            
            return {
                'score': float(max(0.0, min(1.0, overall_score))),
                'rhythm_quality': float(max(0.0, min(1.0, rhythm_score))),
                'intonation_quality': float(max(0.0, min(1.0, intonation_score)))
            }
            
        except Exception as e:
            logger.error(f"French prosody analysis failed: {str(e)}")
            return {'score': 0.0, 'rhythm_quality': 0.0, 'intonation_quality': 0.0}

    def _analyze_intonation_pattern(self, spec: torch.Tensor) -> float:
        """Analyze French intonation patterns."""
        try:
            # Extract pitch contour approximation
            pitch_band = torch.mean(spec[:, 50:150], dim=1)  # Focus on typical pitch range
            
            # Calculate pitch variation
            pitch_variance = torch.std(pitch_band)
            mean_pitch = torch.mean(pitch_band)
            
            # Score based on typical French intonation patterns
            score = 1.0 - (pitch_variance / (mean_pitch + 1e-8))
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Intonation analysis failed: {str(e)}")
            return 0.0

    def _analyze_french_vowels(self, spec: torch.Tensor) -> Dict[str, Any]:
        """Analyze French vowel quality."""
        try:
            # Focus on vowel frequency region
            vowel_region = spec[:, :spec.shape[1]//2]
            
            # Calculate vowel clarity metrics
            clarity_score = float(torch.mean(vowel_region) / (torch.max(vowel_region) + 1e-8))
            
            # Analyze formant structure
            formant_score = self._analyze_formant_structure(vowel_region)
            
            # Calculate overall quality score
            quality_score = (clarity_score + formant_score) / 2
            
            # Generate description
            description = self._describe_vowel_quality(quality_score, formant_score)
            
            return {
                'quality_score': float(max(0.0, min(1.0, quality_score))),
                'formant_structure': float(max(0.0, min(1.0, formant_score))),
                'description': description
            }
            
        except Exception as e:
            logger.error(f"French vowel analysis failed: {str(e)}")
            return {
                'quality_score': 0.0,
                'formant_structure': 0.0,
                'description': "Analysis failed"
            }

    def _analyze_formant_structure(self, vowel_region: torch.Tensor) -> float:
        """Analyze formant structure for French vowels."""
        try:
            peaks = []
            for i in range(1, vowel_region.shape[1]-1):
                if torch.all(vowel_region[:, i] > vowel_region[:, i-1]) and \
                   torch.all(vowel_region[:, i] > vowel_region[:, i+1]):
                    peaks.append(i)
            
            if not peaks:
                return 0.0
            
            # Analyze formant spacing
            formant_gaps = np.diff(peaks)
            typical_gap = 15  # Typical formant spacing for French
            
            # Calculate deviation from typical French formant structure
            gap_scores = [1.0 - abs(gap - typical_gap) / typical_gap for gap in formant_gaps]
            return float(np.mean(gap_scores)) if gap_scores else 0.0
            
        except Exception as e:
            logger.error(f"Formant structure analysis failed: {str(e)}")
            return 0.0

    def _describe_nasal_qualities(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed description of nasal vowel qualities."""
        try:
            descriptions = []
            
            # Assess resonance strength
            strength = analysis['nasal_resonance']['strength']
            if strength > 0.7:
                descriptions.append("Strong nasal resonance")
            elif strength > 0.4:
                descriptions.append("Moderate nasal resonance")
            else:
                descriptions.append("Weak nasal resonance")

            # Assess stability
            stability = analysis['nasal_resonance']['stability']
            if stability < 0.2:
                descriptions.append("Stable nasal resonance")
            elif stability < 0.4:
                descriptions.append("Somewhat stable nasal resonance")
            else:
                descriptions.append("Unstable nasal resonance")

            # Assess distinction
            distinction = analysis['quality_assessment']['distinction']
            if distinction > 0.7:
                descriptions.append("Clear distinction between nasal and oral vowels")
            elif distinction > 0.4:
                descriptions.append("Moderate distinction between nasal and oral vowels")
            else:
                descriptions.append("Limited distinction between nasal and oral vowels")

            return ". ".join(descriptions)
            
        except Exception as e:
            logger.error(f"Nasal quality description failed: {str(e)}")
            return "Unable to analyze nasal qualities"

    def _describe_vowel_quality(self, quality_score: float, formant_score: float) -> str:
        """Generate description of vowel quality."""
        try:
            quality_descriptions = []
            
            # Overall quality assessment
            if quality_score > 0.8:
                quality_descriptions.append("Excellent vowel articulation")
            elif quality_score > 0.6:
                quality_descriptions.append("Good vowel articulation")
            elif quality_score > 0.4:
                quality_descriptions.append("Fair vowel articulation")
            else:
                quality_descriptions.append("Poor vowel articulation")

            # Formant structure assessment
            if formant_score > 0.8:
                quality_descriptions.append("Clear formant structure typical of French vowels")
            elif formant_score > 0.6:
                quality_descriptions.append("Generally good formant structure")
            elif formant_score > 0.4:
                quality_descriptions.append("Inconsistent formant structure")
            else:
                quality_descriptions.append("Unclear formant structure")

            return ". ".join(quality_descriptions)
            
        except Exception as e:
            logger.error(f"Vowel quality description failed: {str(e)}")
            return "Unable to analyze vowel quality"