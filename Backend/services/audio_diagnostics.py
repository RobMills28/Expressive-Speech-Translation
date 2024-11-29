import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioDiagnostics:
    """Diagnostic tool to help identify and address audio quality issues in translations."""
    
    def __init__(self):
        self.quality_metrics = {
            'robotic_voice': 0,      # 1-5 scale (1: very robotic, 5: very natural)
            'pronunciation': 0,       # 1-5 scale (1: poor, 5: excellent)
            'audio_clarity': 0,      # 1-5 scale (1: muddy, 5: crystal clear)
            'background_noise': 0,    # 1-5 scale (1: very noisy, 5: clean)
            'voice_consistency': 0,   # 1-5 scale (1: very inconsistent, 5: very consistent)
        }
        
        self.common_issues = {
            'clipping': False,           # Audio peaks being cut off
            'metallic_artifacts': False, # Metallic/robotic sound artifacts
            'sibilance': False,          # Excessive 's' sounds
            'choppy': False,             # Discontinuous speech
            'muffled': False,            # Lack of high frequencies
            'echo': False,               # Echo/reverb artifacts
        }
        
        self.language_specific_issues = {
            'fra': {
                'nasal_sounds': False,     # Issues with French nasal sounds
                'liaison_breaks': False,    # Incorrect liaison between words
            },
            'deu': {
                'harsh_consonants': False,  # Overly harsh consonants
                'umlauts': False,          # Issues with umlauts
            },
            'spa': {
                'rolling_r': False,        # Issues with rolled 'r' sounds
                'rapid_speech': False,     # Issues with rapid speech segments
            },
            'ita': {
                'consonant_gemination': False,  # Double consonant issues
                'vowel_length': False,          # Incorrect vowel durations
            },
            'por': {
                'nasalization': False,     # Issues with nasal sounds
                'closed_vowels': False,    # Issues with closed vowels
            }
        }

    def analyze_waveform(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze audio waveform characteristics."""
        try:
            # Convert to numpy for analysis
            if isinstance(audio, torch.Tensor):
                audio_np = audio.numpy()
            else:
                audio_np = audio

            analysis = {
                'peak_amplitude': float(np.max(np.abs(audio_np))),
                'avg_amplitude': float(np.mean(np.abs(audio_np))),
                'rms_level': float(np.sqrt(np.mean(audio_np**2))),
                'crest_factor': float(np.max(np.abs(audio_np)) / np.sqrt(np.mean(audio_np**2))) if np.mean(audio_np**2) > 0 else 0,
                'zero_crossings': int(np.sum(np.diff(np.signbit(audio_np)))),
                'silence_percentage': float(np.mean(np.abs(audio_np) < 0.01) * 100),
                'clipping_points': int(np.sum(np.abs(audio_np) > 0.99)),
            }
            
            # Detect potential issues
            self.common_issues['clipping'] = analysis['clipping_points'] > 0
            self.common_issues['muffled'] = analysis['zero_crossings'] < len(audio_np) * 0.1
            self.common_issues['choppy'] = analysis['silence_percentage'] > 10
            
            return analysis
            
        except Exception as e:
            logger.error(f"Waveform analysis failed: {str(e)}")
            return {}

    def analyze_translation(self, audio: torch.Tensor, target_language: str) -> dict:
        """Analyze translation output comprehensively."""
        analysis = {
            'metrics': self.quality_metrics.copy(),
            'issues': self.common_issues.copy(),
            'language_specific': self.language_specific_issues.get(target_language, {}),
            'waveform_analysis': self.analyze_waveform(audio)
        }
        return analysis

    def generate_report(self, analysis: dict, target_language: str) -> str:
        """Generate a detailed report of the analysis."""
        report = [
            "Audio Quality Analysis Report",
            "=" * 30,
            f"\nTarget Language: {target_language}",
            "\nWaveform Analysis:",
            "-" * 20
        ]
        
        # Add waveform metrics
        for metric, value in analysis['waveform_analysis'].items():
            report.append(f"- {metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Add quality metrics
        report.append("\nQuality Metrics (1-5 scale):")
        report.append("-" * 20)
        for metric, value in analysis['metrics'].items():
            report.append(f"- {metric.replace('_', ' ').title()}: {value}")
        
        # Add detected issues
        report.append("\nDetected Issues:")
        report.append("-" * 20)
        for issue, present in analysis['issues'].items():
            if present:
                report.append(f"- {issue.replace('_', ' ').title()}")
        
        # Add language-specific issues
        if analysis['language_specific']:
            report.append("\nLanguage-Specific Issues:")
            report.append("-" * 20)
            for issue, present in analysis['language_specific'].items():
                if present:
                    report.append(f"- {issue.replace('_', ' ').title()}")
        
        return "\n".join(report)