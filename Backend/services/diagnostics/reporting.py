"""
Report generation functionality for audio diagnostics.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates diagnostic reports from analysis results."""

    def __init__(self):
        self.report_dir = Path('diagnostic_reports')
        self.report_dir.mkdir(exist_ok=True)

    def generate_report(self, analysis: dict, target_language: str) -> str:
        """
        Generate a detailed report of the analysis.
        
        Args:
            analysis (dict): Comprehensive analysis results
            target_language (str): Target language code
        
        Returns:
            str: Detailed analysis report
        """
        try:
            if not isinstance(analysis, dict):
                raise ValueError("Invalid analysis data")
            
            if not analysis:
                raise ValueError("Empty analysis dictionary")

            report = [
                "Audio Quality Analysis Report",
                "=" * 30,
                f"\nTarget Language: {target_language}",
                "\nWaveform Analysis:",
                "-" * 20
            ]
        
            # Add waveform metrics if available
            if 'waveform_analysis' in analysis:
                for metric, value in analysis['waveform_analysis'].items():
                    try:
                        if isinstance(value, dict):
                            # Handle nested metrics (processing_metrics, temporal_metrics, volume_metrics)
                            report.append(f"\n- {metric.replace('_', ' ').title()}:")
                            for sub_metric, sub_value in value.items():
                                report.append(f"  - {sub_metric.replace('_', ' ').title()}: {sub_value:.3f}")
                        else:
                            report.append(f"- {metric.replace('_', ' ').title()}: {value:.3f}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Invalid metric value for {metric}: {str(e)}")
                        report.append(f"- {metric.replace('_', ' ').title()}: N/A")
                       
            # Add spectral analysis if available
            if 'spectral_analysis' in analysis:
                report.append("\nSpectral Analysis:")
                report.append("-" * 20)
                
                freq_bands = analysis['spectral_analysis'].get('frequency_bands', {})
                for band, energy in freq_bands.items():
                    try:
                        report.append(f"- {band.title()} Band Energy: {energy:.3f}")
                    except (TypeError, ValueError) as e:
                        report.append(f"- {band.title()} Band Energy: N/A")

            # Add quality metrics
            if 'metrics' in analysis:
                report.append("\nQuality Metrics (1-5 scale):")
                report.append("-" * 20)
                for metric, value in analysis['metrics'].items():
                    try:
                        report.append(f"- {metric.replace('_', ' ').title()}: {float(value)}")
                    except (TypeError, ValueError):
                        report.append(f"- {metric.replace('_', ' ').title()}: N/A")

            # Add detected issues    
            if 'issues' in analysis:
                report.append("\nDetected Issues:")
                report.append("-" * 20)
                detected = False
                for issue, present in analysis['issues'].items():
                    if present:
                        report.append(f"- {issue.replace('_', ' ').title()}")
                        detected = True
                if not detected:
                    report.append("No issues detected")
                    
            # Add language-specific issues
            if 'language_specific' in analysis and analysis['language_specific']:
                report.append("\nLanguage-Specific Analysis:")
                report.append("-" * 20)
                for category, results in analysis['language_specific'].items():
                    report.append(f"\n{category.replace('_', ' ').title()}:")
                    if isinstance(results, dict):
                        for metric, value in results.items():
                            if isinstance(value, (int, float)):
                                report.append(f"  - {metric.replace('_', ' ').title()}: {value:.3f}")
                            elif isinstance(value, bool):
                                report.append(f"  - {metric.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
                            else:
                                report.append(f"  - {metric.replace('_', ' ').title()}: {value}")
                    else:
                        report.append(f"  {results}")

            return "\n".join(report)
        
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return f"Error generating report: {str(e)}"

    def generate_comprehensive_report(
        self,
        audio_analysis: Dict[str, Any],
        target_language: str,
        include_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report with detailed information.
        
        Args:
            audio_analysis: Complete analysis results
            target_language: Target language code
            include_visualizations: Whether to include visualization data
            
        Returns:
            Dict[str, Any]: Comprehensive report
        """
        try:
            # Generate summary scores
            quality_scores = self._calculate_quality_scores(audio_analysis)
            
            # Generate natural language descriptions
            descriptions = {
                'technical_description': self._describe_technical_quality(audio_analysis),
                'perceptual_description': self._describe_perceptual_quality(audio_analysis),
                'linguistic_description': self._describe_linguistic_quality(audio_analysis, target_language),
                'overall_assessment': self._generate_overall_assessment(quality_scores)
            }
            
            # Compile report
            report = {
                'summary': {
                    'quality_scores': quality_scores,
                    'key_findings': self._identify_key_findings(audio_analysis),
                    'recommendations': self._generate_recommendations(quality_scores)
                },
                'detailed_analysis': audio_analysis,
                'descriptions': descriptions,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'target_language': target_language,
                    'analysis_version': '2.0.0'
                }
            }
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_quality_scores(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores from analysis results."""
        try:
            scores = {
                'technical_quality': self._calculate_technical_score(analysis),
                'perceptual_quality': self._calculate_perceptual_score(analysis),
                'linguistic_quality': self._calculate_linguistic_score(analysis)
            }
            
            # Calculate overall quality
            scores['overall_quality'] = sum(scores.values()) / len(scores)
            
            return {k: float(v) for k, v in scores.items()}
        except Exception as e:
            logger.error(f"Quality score calculation failed: {str(e)}")
            return {}

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.report_dir / f"report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Report saving failed: {str(e)}")

    def _describe_technical_quality(self, analysis: Dict[str, Any]) -> str:
        """Generate description of technical quality."""
        try:
            descriptions = []
            
            if 'waveform_analysis' in analysis:
                wa = analysis['waveform_analysis']
                if wa.get('peak_amplitude', 0) > 0.9:
                    descriptions.append("Strong signal level with good amplitude")
                elif wa.get('peak_amplitude', 0) < 0.3:
                    descriptions.append("Low signal level may affect quality")
                    
                if wa.get('clipping_points', 0) > 0:
                    descriptions.append("Some audio clipping detected")
                    
            if 'spectral_analysis' in analysis:
                sa = analysis['spectral_analysis']
                if 'frequency_bands' in sa:
                    fb = sa['frequency_bands']
                    if fb.get('high', 0) < fb.get('mid', 0) * 0.5:
                        descriptions.append("Limited high frequency content")
                    if fb.get('low', 0) > fb.get('mid', 0) * 1.5:
                        descriptions.append("Strong bass presence")
                        
            return ". ".join(descriptions) if descriptions else "No significant technical issues detected"
            
        except Exception as e:
            logger.error(f"Technical quality description failed: {str(e)}")
            return "Error generating technical description"

    def _describe_perceptual_quality(self, analysis: Dict[str, Any]) -> str:
        """Generate description of perceptual quality."""
        try:
            descriptions = []
            
            if 'metrics' in analysis:
                metrics = analysis['metrics']
                
                if metrics.get('robotic_voice', 0) > 3:
                    descriptions.append("Natural-sounding voice quality")
                elif metrics.get('robotic_voice', 0) < 2:
                    descriptions.append("Voice quality shows some artificial characteristics")
                    
                if metrics.get('pronunciation', 0) > 4:
                    descriptions.append("Excellent pronunciation clarity")
                elif metrics.get('pronunciation', 0) < 3:
                    descriptions.append("Pronunciation clarity could be improved")
                    
            return ". ".join(descriptions) if descriptions else "No significant perceptual issues detected"
            
        except Exception as e:
            logger.error(f"Perceptual quality description failed: {str(e)}")
            return "Error generating perceptual description"

    def _describe_linguistic_quality(self, analysis: Dict[str, Any], target_language: str) -> str:
        """Generate description of linguistic quality."""
        try:
            descriptions = []
            
            if 'language_specific' in analysis:
                lang_analysis = analysis['language_specific']
                
                for feature, value in lang_analysis.items():
                    if isinstance(value, dict):
                        if value.get('quality', 0) > 0.8:
                            descriptions.append(f"Excellent {feature.replace('_', ' ')} characteristics")
                        elif value.get('quality', 0) < 0.4:
                            descriptions.append(f"Could improve {feature.replace('_', ' ')}")
                            
            return ". ".join(descriptions) if descriptions else "No significant linguistic issues detected"
            
        except Exception as e:
            logger.error(f"Linguistic quality description failed: {str(e)}")
            return "Error generating linguistic description"

    def _generate_overall_assessment(self, quality_scores: Dict[str, float]) -> str:
        """Generate overall quality assessment."""
        try:
            overall_quality = quality_scores.get('overall_quality', 0)
            
            if overall_quality > 0.8:
                return "Excellent overall quality with strong performance across all aspects"
            elif overall_quality > 0.6:
                return "Good overall quality with some room for improvement"
            elif overall_quality > 0.4:
                return "Fair quality with several areas needing improvement"
            else:
                return "Quality needs significant improvement across multiple areas"
                
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {str(e)}")
            return "Error generating overall assessment"

    def _identify_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify key findings from analysis."""
        try:
            findings = []
            
            # Add significant technical findings
            if analysis.get('waveform_analysis', {}).get('clipping_points', 0) > 0:
                findings.append("Audio clipping detected")
                
            # Add quality metrics findings
            metrics = analysis.get('metrics', {})
            for metric, value in metrics.items():
                if value < 2:
                    findings.append(f"Low {metric.replace('_', ' ')} quality")
                elif value > 4:
                    findings.append(f"Excellent {metric.replace('_', ' ')} quality")
                    
            return findings
            
        except Exception as e:
            logger.error(f"Key findings identification failed: {str(e)}")
            return ["Error identifying key findings"]

    def _generate_recommendations(self, quality_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on quality scores."""
        try:
            recommendations = []
            
            for aspect, score in quality_scores.items():
                if score < 0.4:
                    recommendations.append(
                        f"Priority: Improve {aspect.replace('_', ' ')} - "
                        f"Current score: {score:.2f}"
                    )
                elif score < 0.7:
                    recommendations.append(
                        f"Consider improving {aspect.replace('_', ' ')} - "
                        f"Current score: {score:.2f}"
                    )
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {str(e)}")
            return ["Error generating recommendations"]