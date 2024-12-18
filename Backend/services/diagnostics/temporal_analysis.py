"""
Temporal analysis functionality for audio signal processing.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Analyzes temporal characteristics of audio signals."""

    def analyze_temporal_structure(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze temporal characteristics with focus on speech-relevant features.
        """
        try:
            audio_np = audio.squeeze().numpy()
            
            # Envelope analysis
            analytic_signal = np.abs(audio_np)
            envelope = np.abs(analytic_signal)
            
            # Detect speech segments and pauses
            is_speech = envelope > np.mean(envelope) * 0.1
            speech_segments = self._find_continuous_segments(is_speech)
            
            # Temporal pattern analysis
            temporal_data = {
                'speech_segments': {
                    'count': len(speech_segments),
                    'mean_duration': float(np.mean([s[1] - s[0] for s in speech_segments])) if speech_segments else 0.0,
                    'duration_variance': float(np.var([s[1] - s[0] for s in speech_segments])) if speech_segments else 0.0,
                    'pattern_description': self._describe_temporal_patterns(speech_segments)
                },
                'rhythm_analysis': {
                    'speech_rate': self._estimate_speech_rate(speech_segments),
                    'rhythm_regularity': self._analyze_rhythm_regularity(speech_segments),
                    'pause_patterns': self._analyze_pause_patterns(speech_segments)
                },
                'energy_dynamics': {
                    'attack_characteristics': self._analyze_attacks(envelope),
                    'decay_characteristics': self._analyze_decays(envelope),
                    'sustain_patterns': self._analyze_sustain_patterns(envelope),
                    'dynamic_range': float(np.log10(np.max(envelope) / (np.min(envelope) + 1e-8)))
                }
            }
            
            return temporal_data
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {str(e)}")
            return {}

    def _find_continuous_segments(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments in a boolean signal."""
        try:
            segments = []
            start = None
            
            for i, val in enumerate(signal):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    segments.append((start, i))
                    start = None
                    
            if start is not None:
                segments.append((start, len(signal)))
                
            return segments
            
        except Exception as e:
            logger.error(f"Segment finding failed: {str(e)}")
            return []

    def _describe_temporal_patterns(self, segments: List[Tuple[int, int]]) -> str:
        """Generate natural language description of temporal patterns."""
        try:
            if not segments:
                return "No clear speech segments detected"
                
            durations = [s[1] - s[0] for s in segments]
            mean_dur = np.mean(durations)
            var_dur = np.var(durations)
            
            description = []
            if len(segments) < 5:
                description.append("Speech contains very few segments, indicating potential choppy or interrupted speech")
            elif var_dur / mean_dur > 0.5:
                description.append("Highly variable segment durations suggest irregular speech patterns")
            else:
                description.append("Consistent segment durations indicate smooth and regular speech flow")
                
            # Analyze spacing between segments
            gaps = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
            if gaps:
                mean_gap = np.mean(gaps)
                if mean_gap > 0.5 * mean_dur:
                    description.append("Long pauses between speech segments may indicate unnatural pacing")
                elif mean_gap < 0.1 * mean_dur:
                    description.append("Very short gaps between segments suggest rushed or connected speech")
                    
            return ". ".join(description)
            
        except Exception as e:
            logger.error(f"Pattern description failed: {str(e)}")
            return "Error analyzing temporal patterns"

    def _estimate_speech_rate(self, segments: List[Tuple[int, int]]) -> float:
        """Estimate speech rate from segments."""
        try:
            if not segments:
                return 0.0
            total_speech_time = sum(s[1] - s[0] for s in segments)
            total_time = segments[-1][1] - segments[0][0]
            return float(len(segments) / (total_time / 16000))  # Assuming 16kHz sample rate
        except Exception as e:
            logger.error(f"Speech rate estimation failed: {str(e)}")
            return 0.0

    def _analyze_rhythm_regularity(self, segments: List[Tuple[int, int]]) -> float:
        """Analyze rhythm regularity from segments."""
        try:
            if len(segments) < 2:
                return 0.0
            intervals = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
            return float(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8))
        except Exception as e:
            logger.error(f"Rhythm regularity analysis failed: {str(e)}")
            return 0.0

    def _analyze_pause_patterns(self, segments: List[Tuple[int, int]]) -> Dict[str, float]:
        """Analyze pause patterns between segments."""
        try:
            if len(segments) < 2:
                return {'mean_pause': 0.0, 'pause_variability': 0.0}
            
            pauses = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
            return {
                'mean_pause': float(np.mean(pauses)),
                'pause_variability': float(np.std(pauses) / (np.mean(pauses) + 1e-8)),
                'min_pause': float(np.min(pauses)),
                'max_pause': float(np.max(pauses)),
                'pause_count': len(pauses)
            }
        except Exception as e:
            logger.error(f"Pause pattern analysis failed: {str(e)}")
            return {
                'mean_pause': 0.0,
                'pause_variability': 0.0,
                'min_pause': 0.0,
                'max_pause': 0.0,
                'pause_count': 0
            }

    def _analyze_attacks(self, envelope: np.ndarray) -> Dict[str, float]:
        """Analyze attack characteristics of the audio envelope."""
        try:
            # Find attack regions (rising edges)
            diff = np.diff(envelope)
            attack_regions = np.where(diff > np.mean(diff) + np.std(diff))[0]
            
            if len(attack_regions) == 0:
                return {
                    'attack_count': 0,
                    'mean_attack_time': 0.0,
                    'attack_strength': 0.0
                }
            
            # Analyze attack characteristics
            attack_times = []
            attack_strengths = []
            
            for i in range(len(attack_regions) - 1):
                if attack_regions[i + 1] - attack_regions[i] > 1:
                    # Found a complete attack region
                    region = envelope[attack_regions[i]:attack_regions[i + 1]]
                    attack_times.append(len(region))
                    attack_strengths.append(np.max(region) - np.min(region))
            
            return {
                'attack_count': len(attack_times),
                'mean_attack_time': float(np.mean(attack_times)) if attack_times else 0.0,
                'attack_strength': float(np.mean(attack_strengths)) if attack_strengths else 0.0
            }
            
        except Exception as e:
            logger.error(f"Attack analysis failed: {str(e)}")
            return {
                'attack_count': 0,
                'mean_attack_time': 0.0,
                'attack_strength': 0.0
            }

    def _analyze_decays(self, envelope: np.ndarray) -> Dict[str, float]:
        """Analyze decay characteristics of the audio envelope."""
        try:
            # Find decay regions (falling edges)
            diff = np.diff(envelope)
            decay_regions = np.where(diff < np.mean(diff) - np.std(diff))[0]
            
            if len(decay_regions) == 0:
                return {
                    'decay_count': 0,
                    'mean_decay_time': 0.0,
                    'decay_smoothness': 0.0
                }
            
            # Analyze decay characteristics
            decay_times = []
            decay_smoothness = []
            
            for i in range(len(decay_regions) - 1):
                if decay_regions[i + 1] - decay_regions[i] > 1:
                    # Found a complete decay region
                    region = envelope[decay_regions[i]:decay_regions[i + 1]]
                    decay_times.append(len(region))
                    # Measure smoothness as correlation with ideal exponential decay
                    ideal_decay = np.exp(-np.linspace(0, 5, len(region)))
                    correlation = np.corrcoef(region, ideal_decay)[0, 1]
                    decay_smoothness.append(max(0, correlation))
            
            return {
                'decay_count': len(decay_times),
                'mean_decay_time': float(np.mean(decay_times)) if decay_times else 0.0,
                'decay_smoothness': float(np.mean(decay_smoothness)) if decay_smoothness else 0.0
            }
            
        except Exception as e:
            logger.error(f"Decay analysis failed: {str(e)}")
            return {
                'decay_count': 0,
                'mean_decay_time': 0.0,
                'decay_smoothness': 0.0
            }

    def _analyze_sustain_patterns(self, envelope: np.ndarray) -> Dict[str, float]:
        """Analyze sustain patterns in the audio envelope."""
        try:
            # Define sustain threshold as regions with relatively stable amplitude
            mean_amplitude = np.mean(envelope)
            amplitude_deviation = np.std(envelope)
            sustain_threshold = amplitude_deviation * 0.25
            
            # Find sustained regions
            is_sustained = np.abs(envelope - mean_amplitude) < sustain_threshold
            sustained_segments = self._find_continuous_segments(is_sustained)
            
            if not sustained_segments:
                return {
                    'sustain_count': 0,
                    'mean_sustain_length': 0.0,
                    'sustain_stability': 0.0,
                    'total_sustain_ratio': 0.0
                }
            
            # Calculate sustain characteristics
            sustain_lengths = [s[1] - s[0] for s in sustained_segments]
            sustain_stability = []
            
            for start, end in sustained_segments:
                if end - start > 1:
                    region = envelope[start:end]
                    stability = 1.0 - (np.std(region) / (np.mean(region) + 1e-8))
                    sustain_stability.append(stability)
            
            total_sustain_time = sum(sustain_lengths)
            
            return {
                'sustain_count': len(sustained_segments),
                'mean_sustain_length': float(np.mean(sustain_lengths)),
                'sustain_stability': float(np.mean(sustain_stability)) if sustain_stability else 0.0,
                'total_sustain_ratio': float(total_sustain_time / len(envelope))
            }
            
        except Exception as e:
            logger.error(f"Sustain pattern analysis failed: {str(e)}")
            return {
                'sustain_count': 0,
                'mean_sustain_length': 0.0,
                'sustain_stability': 0.0,
                'total_sustain_ratio': 0.0
            }