"""
Base AudioDiagnostics class with core functionality.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import numpy as np

from .quality_metrics import AudioQualityLevel
from .spectral_analysis import SpectralAnalyzer
from .temporal_analysis import TemporalAnalyzer
from .language_analysis import (
    FrenchAnalyzer,
    GermanAnalyzer,
    ItalianAnalyzer,
    PortugueseAnalyzer,
    SpanishAnalyzer
)
from .reporting import ReportGenerator
from .utils import (
    calculate_spectral_slope,
    analyze_peak_regularity,
    calculate_frequency_bands,
    analyze_temporal_stability
)

logger = logging.getLogger(__name__)

class AudioDiagnostics:
    """Core audio diagnostics functionality."""

    def __init__(self):
        """Initialize diagnostic components."""
        try:
            # Initialize analyzers
            self.spectral_analyzer = SpectralAnalyzer()
            self.temporal_analyzer = TemporalAnalyzer()
            self.report_generator = ReportGenerator()
            
            # Initialize language analyzers
            self.language_analyzers = {
                'fra': FrenchAnalyzer(),
                'deu': GermanAnalyzer(),
                'ita': ItalianAnalyzer(),
                'por': PortugueseAnalyzer(),
                'spa': SpanishAnalyzer()
            }

            # Initialize quality metrics
            self.quality_metrics = {
                'robotic_voice': 0.0,
                'pronunciation': 0.0,
                'audio_clarity': 0.0,
                'background_noise': 0.0,
                'voice_consistency': 0.0,
                'spectral_balance': 0.0
            }

            # Initialize common issues tracking
            self.common_issues = {
                'clipping': False,
                'metallic_artifacts': False,
                'sibilance': False,
                'choppy': False,
                'muffled': False,
                'echo': False,
                'spectral_imbalance': False
            }

            # Initialize language-specific issues
            self.language_specific_issues = {}

            logger.info("AudioDiagnostics initialized successfully")
        except Exception as e:
            logger.error(f"AudioDiagnostics initialization failed: {str(e)}")
            raise

    def analyze_translation(self, audio: torch.Tensor, target_language: str) -> dict:
        """
        Main analysis entry point.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            target_language (str): Target language code
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            # Basic validation
            if not isinstance(audio, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")

            if audio.dim() == 0 or audio.size(0) == 0:
                raise ValueError("Empty audio tensor")

            # Perform comprehensive analysis
            waveform_analysis = self.analyze_waveform(audio)
            spectral_analysis = self.spectral_analyzer.analyze_spectral_characteristics(audio)
            temporal_analysis = self.temporal_analyzer.analyze_temporal_structure(audio)
            
            # Language-specific analysis
            language_analysis = {}
            if target_language in self.language_analyzers:
                try:
                    analyzer = self.language_analyzers[target_language]
                    language_analysis = analyzer.analyze(audio)
                    # Update language-specific issues
                    self.language_specific_issues[target_language] = language_analysis
                    logger.info(f"Completed language analysis for {target_language}")
                except Exception as e:
                    logger.error(f"Language analysis failed for {target_language}: {str(e)}")
                    language_analysis = {}
            else:
                logger.warning(f"No analyzer available for language: {target_language}")

            # Update quality metrics based on analyses
            self._update_quality_metrics(
                waveform_analysis,
                spectral_analysis,
                temporal_analysis,
                language_analysis
            )

            # Compile final analysis
            analysis = {
                'metrics': self.quality_metrics.copy(),
                'issues': self.common_issues.copy(),
                'language_specific': self.language_specific_issues.get(target_language, {}),
                'waveform_analysis': waveform_analysis,
                'spectral_analysis': {
                    'frequency_bands': spectral_analysis.get('frequency_bands', {}),
                    'characteristics': spectral_analysis.get('overall_characteristics', {})
                },
                'temporal_analysis': temporal_analysis,
                'language_analysis': language_analysis
            }

            # Add timestamp and metadata
            analysis['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'target_language': target_language,
                'version': '2.0.0'
            }

            return analysis

        except Exception as e:
            logger.error(f"Translation analysis failed: {str(e)}")
            return self._generate_error_analysis()

    def analyze_waveform(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze audio waveform characteristics with enhanced processing detection."""
        try:
            # Convert to numpy for analysis if needed
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().numpy()
            else:
                audio_np = audio

            # Basic waveform metrics
            analysis = {
                'peak_amplitude': float(np.max(np.abs(audio_np))),
                'avg_amplitude': float(np.mean(np.abs(audio_np))),
                'rms_level': float(np.sqrt(np.mean(audio_np**2))),
                'crest_factor': float(np.max(np.abs(audio_np)) / (np.sqrt(np.mean(audio_np**2)) + 1e-8)),
                'zero_crossings': int(np.sum(np.diff(np.signbit(audio_np)))),
                'silence_percentage': float(np.mean(np.abs(audio_np) < 0.001) * 100),
                'clipping_points': int(np.sum(np.abs(audio_np) > 0.99))
            }

            # Enhanced dynamic range analysis
            sorted_amplitudes = np.sort(np.abs(audio_np))
            percentile_90 = np.percentile(sorted_amplitudes, 90)
            percentile_10 = np.percentile(sorted_amplitudes, 10)
            analysis['dynamic_range'] = float(percentile_90 / (percentile_10 + 1e-8))
        
            # Overprocessing detection
            analysis['processing_metrics'] = {
                'compression_detected': analysis['crest_factor'] < 4.0,
                'over_compressed': analysis['dynamic_range'] < 10.0,
                'compression_severity': max(0.0, 1.0 - (analysis['crest_factor'] / 8.0)),
                'dynamic_range_loss': max(0.0, 1.0 - (analysis['dynamic_range'] / 20.0))
            }

            # Temporal variation analysis
            temporal_variations = np.diff(audio_np)
            analysis['temporal_metrics'] = {
                'smoothing_detected': float(np.std(temporal_variations)) < 0.1,
                'temporal_variation': float(np.std(temporal_variations)),
                'temporal_flatness': float(np.mean(np.abs(temporal_variations)))
            }

            # Volume consistency analysis
            frame_size = 1024
            frames = [audio_np[i:i+frame_size] for i in range(0, len(audio_np)-frame_size, frame_size)]
            if frames:
                frame_rms = np.array([np.sqrt(np.mean(frame**2)) for frame in frames])
                analysis['volume_metrics'] = {
                    'volume_stability': float(1.0 - np.std(frame_rms) / (np.mean(frame_rms) + 1e-8)),
                    'volume_range': float(np.max(frame_rms) / (np.min(frame_rms) + 1e-8)),
                    'volume_variations': float(np.std(frame_rms))
                }
            else:
                analysis['volume_metrics'] = {
                    'volume_stability': 0.0,
                    'volume_range': 0.0,
                    'volume_variations': 0.0
                }

            # Update common issues based on enhanced analysis
            self._update_common_issues(analysis)
        
            return analysis
        
        except Exception as e:
            logger.error(f"Waveform analysis failed: {str(e)}")
            return {}

    def generate_report(self, analysis: dict, target_language: str) -> str:
        """Generate analysis report."""
        try:
            return self.report_generator.generate_report(analysis, target_language)
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return f"Error generating report: {str(e)}"

    def _update_quality_metrics(
        self,
        waveform_analysis: Dict[str, Any],
        spectral_analysis: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
        language_analysis: Dict[str, Any]
    ) -> None:
        """Update quality metrics based on analyses."""
        try:
            # Update robotic voice metric
            self.quality_metrics['robotic_voice'] = self._calculate_robotic_score(
                spectral_analysis, temporal_analysis
            )
            
            # Update pronunciation metric
            self.quality_metrics['pronunciation'] = self._calculate_pronunciation_score(
                spectral_analysis, language_analysis
            )
            
            # Update audio clarity metric
            self.quality_metrics['audio_clarity'] = self._calculate_clarity_score(
                waveform_analysis, spectral_analysis
            )
            
            # Update background noise metric
            self.quality_metrics['background_noise'] = self._calculate_noise_score(
                waveform_analysis
            )
            
            # Update voice consistency metric
            self.quality_metrics['voice_consistency'] = self._calculate_consistency_score(
                temporal_analysis
            )
            
            # Update spectral balance metric
            self.quality_metrics['spectral_balance'] = self._calculate_balance_score(
                spectral_analysis.get('frequency_bands', {})
            )
            
        except Exception as e:
            logger.error(f"Quality metrics update failed: {str(e)}")

    def _update_common_issues(self, waveform_analysis: Dict[str, Any]) -> None:
        """Update common issues based on waveform analysis."""
        try:
            # Check for clipping
            self.common_issues['clipping'] = waveform_analysis.get('clipping_points', 0) > 0
            
            # Check for low signal level
            self.common_issues['muffled'] = waveform_analysis.get('rms_level', 1.0) < 0.1
            
        except Exception as e:
            logger.error(f"Common issues update failed: {str(e)}")

    def _generate_error_analysis(self) -> Dict[str, Any]:
        """Generate empty analysis for error cases."""
        return {
            'metrics': {metric: 0.0 for metric in self.quality_metrics},
            'issues': {issue: False for issue in self.common_issues},
            'language_specific': {},
            'waveform_analysis': {},
            'spectral_analysis': {'frequency_bands': {}, 'characteristics': {}},
            'temporal_analysis': {},
            'language_analysis': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
        }

    def _calculate_robotic_score(self, spectral_analysis: Dict[str, Any], 
                               temporal_analysis: Dict[str, Any]) -> float:
        """Calculate roboticness score based on spectral and temporal characteristics."""
        try:
            # Get spectral characteristics
            spectral_flatness = spectral_analysis.get('characteristics', {}).get('spectral_flatness', 0.0)
            spectral_centroid = spectral_analysis.get('characteristics', {}).get('spectral_centroid', 0.0)
            
            # Get temporal characteristics
            temporal_regularity = temporal_analysis.get('rhythm_analysis', {}).get('rhythm_regularity', 0.0)
            
            # Combine scores (less mechanical is better)
            score = 5.0 * (1.0 - (spectral_flatness + (1.0 - temporal_regularity)) / 2)
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Robotic score calculation failed: {str(e)}")
            return 1.0

    def _calculate_pronunciation_score(self, spectral_analysis: Dict[str, Any], 
                                    language_analysis: Dict[str, Any]) -> float:
        """Calculate pronunciation score based on spectral clarity and language features."""
        try:
            # Get spectral clarity
            spectral_clarity = spectral_analysis.get('characteristics', {}).get('spectral_centroid', 0.0)
            
            # Get language-specific pronunciation quality
            language_quality = language_analysis.get('overall_quality', 0.0)
            
            # Combine scores
            score = 1.0 + 4.0 * (spectral_clarity + language_quality) / 2
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Pronunciation score calculation failed: {str(e)}")
            return 1.0

    def _calculate_clarity_score(self, waveform_analysis: Dict[str, Any], 
                               spectral_analysis: Dict[str, Any]) -> float:
        """Calculate clarity score based on signal characteristics."""
        try:
            # Get waveform characteristics
            rms_level = waveform_analysis.get('rms_level', 0.0)
            crest_factor = waveform_analysis.get('crest_factor', 1.0)
            
            # Get spectral characteristics
            spectral_spread = spectral_analysis.get('characteristics', {}).get('spectral_spread', 1.0)
            
            # Calculate clarity score
            signal_quality = rms_level / (crest_factor + 1e-8)
            spectral_quality = 1.0 / (spectral_spread + 1e-8)
            
            score = 1.0 + 4.0 * (signal_quality + spectral_quality) / 2
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Clarity score calculation failed: {str(e)}")
            return 1.0

    def _calculate_noise_score(self, waveform_analysis: Dict[str, Any]) -> float:
        """Calculate noise score based on signal characteristics."""
        try:
            rms_level = waveform_analysis.get('rms_level', 0.0)
            peak_amplitude = waveform_analysis.get('peak_amplitude', 0.0)
        
            # Use peak-to-RMS ratio to better handle soft voices
            dynamic_range = peak_amplitude / (rms_level + 1e-8)
        
            # Adjust score based on dynamic range rather than just SNR
            score = 1.0 + 4.0 * min(dynamic_range / 10.0, 1.0)
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Noise score calculation failed: {str(e)}")
            return 1.0

    def _calculate_consistency_score(self, temporal_analysis: Dict[str, Any]) -> float:
        """Calculate voice consistency score based on temporal characteristics."""
        try:
            # Get temporal characteristics
            rhythm_regularity = temporal_analysis.get('rhythm_analysis', {}).get('rhythm_regularity', 0.0)
            energy_stability = temporal_analysis.get('energy_dynamics', {}).get('sustain_stability', 0.0)
            
            # Combine scores
            score = 1.0 + 4.0 * (rhythm_regularity + energy_stability) / 2
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Consistency score calculation failed: {str(e)}")
            return 1.0

    def _calculate_balance_score(self, frequency_bands: Dict[str, float]) -> float:
        """Calculate spectral balance score based on frequency band distribution."""
        try:
            if not frequency_bands:
                return 1.0
                
            # Calculate total energy
            total_energy = sum(frequency_bands.values())
            if total_energy == 0:
                return 1.0
                
            # Calculate relative energies
            ratios = {k: v/total_energy for k, v in frequency_bands.items()}
            
            # Ideal ratios for speech
            ideal_ratios = {
                'low': 0.3,
                'mid': 0.5,
                'high': 0.2
            }
            
            # Calculate deviation from ideal
            deviation = sum(abs(ratios.get(k, 0) - v) for k, v in ideal_ratios.items())
            
            # Convert to 1-5 score (lower deviation is better)
            score = 5.0 * (1.0 - deviation)
            return float(max(1.0, min(5.0, score)))
        except Exception as e:
            logger.error(f"Balance score calculation failed: {str(e)}")
            return 1.0

    def diagnose_translation_quality(
        self,
        source_audio: torch.Tensor,
        translated_audio: torch.Tensor,
        target_language: str
    ) -> Dict[str, Any]:
        """
        Comprehensive diagnosis of translation quality with detailed comparisons.
        
        Args:
            source_audio: Original audio tensor
            translated_audio: Translated audio tensor
            target_language: Target language code
            
        Returns:
            Dict[str, Any]: Comprehensive diagnosis
        """
        try:
            # Analyze both source and translated audio
            source_analysis = self.analyze_translation(source_audio, 'eng')
            translated_analysis = self.analyze_translation(translated_audio, target_language)
            
            # Compare characteristics
            comparison = self._compare_audio_characteristics(
                source_audio,
                translated_audio,
                source_analysis,
                translated_analysis
            )
            
            # Generate comprehensive diagnosis
            diagnosis = {
                'source_analysis': source_analysis,
                'translation_analysis': translated_analysis,
                'comparative_analysis': comparison,
                'recommendations': self._generate_quality_recommendations(comparison)
            }
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Translation quality diagnosis failed: {str(e)}")
            return {}

    def _compare_audio_characteristics(
        self,
        source_audio: torch.Tensor,
        translated_audio: torch.Tensor,
        source_analysis: Dict[str, Any],
        translated_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare characteristics between source and translated audio."""
        try:
            comparison = {
                'duration_ratio': len(translated_audio) / len(source_audio),
                'energy_ratio': (
                    translated_analysis['waveform_analysis'].get('rms_level', 0) /
                    (source_analysis['waveform_analysis'].get('rms_level', 1) + 1e-8)
                ),
                'spectral_differences': self._compare_spectral_characteristics(
                    source_analysis['spectral_analysis'],
                    translated_analysis['spectral_analysis']
                ),
                'quality_comparison': {
                    metric: translated_analysis['metrics'].get(metric, 0) -
                           source_analysis['metrics'].get(metric, 0)
                    for metric in self.quality_metrics
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Audio characteristics comparison failed: {str(e)}")
            return {}

    def _compare_spectral_characteristics(self, 
                                       source_analysis: Dict[str, Any],
                                       translated_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Compare spectral characteristics between source and translated audio."""
        try:
            source_bands = source_analysis.get('frequency_bands', {})
            translated_bands = translated_analysis.get('frequency_bands', {})
            
            if not source_bands or not translated_bands:
                return {}
                
            differences = {}
            for band in set(source_bands.keys()) & set(translated_bands.keys()):
                source_energy = source_bands[band]
                translated_energy = translated_bands[band]
                if source_energy > 0:
                    differences[band] = translated_energy / source_energy - 1.0
                else:
                    differences[band] = 0.0
                    
            return differences
            
        except Exception as e:
            logger.error(f"Spectral characteristics comparison failed: {str(e)}")
            return {}

    def _generate_quality_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations based on comparison."""
        try:
            recommendations = []
            
            # Check duration ratio
            if comparison.get('duration_ratio', 1) > 1.5:
                recommendations.append(
                    "Translation is significantly longer than source. Consider condensing content."
                )
            elif comparison.get('duration_ratio', 1) < 0.5:
                recommendations.append(
                    "Translation is significantly shorter than source. Check for content loss."
                )
                
            # Check energy levels
            if comparison.get('energy_ratio', 1) < 0.7:
                recommendations.append(
                    "Translation audio level is low compared to source. Consider normalization."
                )
                
            # Check quality metrics
            for metric, diff in comparison.get('quality_comparison', {}).items():
                if diff < -0.5:
                    recommendations.append(
                        f"Translation shows degraded {metric.replace('_', ' ')}. "
                        "Consider quality improvement."
                    )
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Quality recommendations generation failed: {str(e)}")
            return ["Error generating recommendations"]