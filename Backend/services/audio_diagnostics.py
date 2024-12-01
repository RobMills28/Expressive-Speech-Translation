import logging
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioDiagnostics:
    """Diagnostic tool to help identify and address audio quality issues in translations."""
    
    def __init__(self):
        # Create diagnostics directory
        Path("diagnostics").mkdir(exist_ok=True)
        
        self.quality_metrics = {
            'robotic_voice': 0,      # 1-5 scale (1: very robotic, 5: very natural)
            'pronunciation': 0,       # 1-5 scale (1: poor, 5: excellent)
            'audio_clarity': 0,      # 1-5 scale (1: muddy, 5: crystal clear)
            'background_noise': 0,    # 1-5 scale (1: very noisy, 5: clean)
            'voice_consistency': 0,   # 1-5 scale (1: very inconsistent, 5: very consistent),
            'spectral_balance': 0     # 1-5 scale (1: poor balance, 5: excellent balance)
        }
        
        self.spectral_metrics = {
            'low_band_energy': 0.0,
            'mid_band_energy': 0.0,
            'high_band_energy': 0.0,
            'band_balance_score': 0.0
        }
        
        self.common_issues = {
            'clipping': False,           # Audio peaks being cut off
            'metallic_artifacts': False, # Metallic/robotic sound artifacts
            'sibilance': False,          # Excessive 's' sounds
            'choppy': False,             # Discontinuous speech
            'muffled': False,            # Lack of high frequencies
            'echo': False,               # Echo/reverb artifacts
            'spectral_imbalance': False  # Poor frequency distribution
        }
        
        self.language_specific_issues = {
            'fra': {
                'nasal_sounds': False,     # Issues with French nasal sounds
                'liaison_breaks': False,    # Incorrect liaison between words
                'vowel_clarity': False,     # Issues with vowel pronunciation
                'prosody': False           # Issues with intonation patterns
            },
            'deu': {
                'harsh_consonants': False,  # Overly harsh consonants
                'umlauts': False,          # Issues with umlauts
                'consonant_clusters': False, # Issues with consonant combinations
                'word_stress': False        # Incorrect stress patterns
            },
            'spa': {
                'rolling_r': False,        # Issues with rolled 'r' sounds
                'rapid_speech': False,     # Issues with rapid speech segments
                'vowel_distinction': False, # Clear distinction between vowels
                'consonant_clarity': False  # Clear consonant pronunciation
            },
            'ita': {
                'consonant_gemination': False,  # Double consonant issues
                'vowel_length': False,          # Incorrect vowel durations
                'stress_patterns': False,       # Issues with word stress
                'rhythm': False                 # Issues with rhythmic patterns
            },
            'por': {
                'nasalization': False,     # Issues with nasal sounds
                'closed_vowels': False,    # Issues with closed vowels
                'consonant_softening': False, # Issues with soft consonants
                'word_endings': False      # Issues with word-final sounds
            }
        }
        
        # Store step-by-step diagnostics
        self.step_diagnostics = {}

    def visualize_waveform(self, audio: torch.Tensor, title: str = "Waveform", save_path: Optional[str] = None) -> None:
        """Visualize audio waveform"""
        plt.figure(figsize=(12, 4))
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().numpy()
        else:
            audio_np = audio
            
        plt.plot(audio_np)
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        if save_path is None:
            save_path = f"diagnostics/{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def visualize_spectrogram(self, 
                            audio: torch.Tensor, 
                            sample_rate: int = 16000, 
                            title: str = "Spectrogram",
                            save_path: Optional[str] = None) -> None:
        """Visualize audio spectrogram"""
        plt.figure(figsize=(12, 8))
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().numpy()
        else:
            audio_np = audio
            
        spec = torch.stft(
            torch.from_numpy(audio_np) if isinstance(audio_np, np.ndarray) else audio,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=torch.hann_window(2048),
            return_complex=True
        )
        spec_mag = torch.abs(spec).numpy()
        
        plt.imshow(
            20 * np.log10(spec_mag + 1e-10),
            aspect='auto',
            origin='lower',
            extent=[0, len(audio_np)/sample_rate, 0, sample_rate/2]
        )
        plt.colorbar(label='dB')
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        
        if save_path is None:
            save_path = f"diagnostics/{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def analyze_waveform(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze audio waveform characteristics."""
        try:
            # Convert to numpy for analysis
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().numpy()
            else:
                audio_np = audio

            analysis = {
                'peak_amplitude': float(np.max(np.abs(audio_np))),
                'avg_amplitude': float(np.mean(np.abs(audio_np))),
                'rms_level': float(np.sqrt(np.mean(audio_np**2))),
                'crest_factor': float(np.max(np.abs(audio_np)) / (np.sqrt(np.mean(audio_np**2)) + 1e-8)),
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

    def analyze_spectral_balance(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze spectral balance of the audio"""
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
            
            # Calculate energy in different frequency bands
            freq_bands = {
                'low': float(torch.mean(mag[:, :mag.shape[1]//4])),
                'mid': float(torch.mean(mag[:, mag.shape[1]//4:mag.shape[1]//2])),
                'high': float(torch.mean(mag[:, mag.shape[1]//2:]))
            }
            
            # Calculate balance score
            total_energy = sum(freq_bands.values())
            if total_energy > 0:
                balance_score = min(
                    freq_bands['mid'] / total_energy,
                    (freq_bands['low'] + freq_bands['high']) / (2 * total_energy)
                )
            else:
                balance_score = 0.0
            
            # Update spectral metrics
            self.spectral_metrics.update({
                'low_band_energy': freq_bands['low'],
                'mid_band_energy': freq_bands['mid'],
                'high_band_energy': freq_bands['high'],
                'band_balance_score': float(balance_score)
            })

            # Update quality metrics based on spectral balance
            self.quality_metrics['spectral_balance'] = int(balance_score * 5)
            self.common_issues['spectral_imbalance'] = balance_score < 0.6
            
            return freq_bands
            
        except Exception as e:
            logger.error(f"Spectral balance analysis failed: {str(e)}")
            return {'low': 0.0, 'mid': 0.0, 'high': 0.0}

    def analyze_translation(self, audio: torch.Tensor, target_language: str) -> dict:
        """Analyze translation output comprehensively."""
        try:
            # Basic waveform analysis
            waveform_analysis = self.analyze_waveform(audio)
            
            # Spectral analysis
            spectral_bands = self.analyze_spectral_balance(audio)
            
            # Language-specific analysis
            if target_language in self.language_specific_issues:
                self._analyze_language_specific(audio, target_language, spectral_bands)
            
            # Compile complete analysis
            analysis = {
                'metrics': self.quality_metrics.copy(),
                'issues': self.common_issues.copy(),
                'language_specific': self.language_specific_issues.get(target_language, {}),
                'waveform_analysis': waveform_analysis,
                'spectral_analysis': {
                    'frequency_bands': spectral_bands,
                    'spectral_metrics': self.spectral_metrics
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Translation analysis failed: {str(e)}")
            return {
                'metrics': self.quality_metrics.copy(),
                'issues': self.common_issues.copy(),
                'language_specific': {},
                'waveform_analysis': {},
                'spectral_analysis': {'frequency_bands': {}, 'spectral_metrics': {}}
            }

    def _analyze_language_specific(self, audio: torch.Tensor, target_language: str, spectral_bands: Dict[str, float]) -> None:
        """Analyze language-specific characteristics"""
        try:
            if target_language == 'fra':
                self.language_specific_issues['fra']['nasal_sounds'] = spectral_bands['mid'] < 0.3
                self.language_specific_issues['fra']['vowel_clarity'] = spectral_bands['low'] < 0.25
            elif target_language == 'deu':
                self.language_specific_issues['deu']['harsh_consonants'] = spectral_bands['high'] > 0.4
                self.language_specific_issues['deu']['consonant_clusters'] = self.spectral_metrics['band_balance_score'] < 0.5
            elif target_language == 'spa':
                self.language_specific_issues['spa']['rolling_r'] = spectral_bands['mid'] < 0.3
                self.language_specific_issues['spa']['consonant_clarity'] = spectral_bands['high'] < 0.2
            elif target_language == 'ita':
                self.language_specific_issues['ita']['consonant_gemination'] = spectral_bands['high'] < 0.25
                self.language_specific_issues['ita']['vowel_length'] = spectral_bands['low'] < 0.3
            elif target_language == 'por':
                self.language_specific_issues['por']['nasalization'] = spectral_bands['mid'] < 0.35
                self.language_specific_issues['por']['closed_vowels'] = spectral_bands['low'] < 0.28
        except Exception as e:
            logger.error(f"Language-specific analysis failed: {str(e)}")

    def analyze_processing_step(
        self, 
        audio: torch.Tensor,
        step_name: str,
        prev_audio: Optional[torch.Tensor] = None,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """Analyze audio at each processing step"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_analysis = {
                'waveform': self.analyze_waveform(audio),
                'spectral': self.analyze_spectral_balance(audio),
                'timestamp': timestamp
            }
            
            self.step_diagnostics[step_name] = step_analysis
            
            if visualize:
                base_path = f"diagnostics/{timestamp}_{step_name.lower().replace(' ', '_')}"
                self.visualize_waveform(
                    audio, 
                    title=f"{step_name} - Waveform",
                    save_path=f"{base_path}_waveform.png"
                )
                self.visualize_spectrogram(
                    audio,
                    title=f"{step_name} - Spectrogram",
                    save_path=f"{base_path}_spectrogram.png"
                )
                
                if prev_audio is not None:
                    self._visualize_comparison(
                        prev_audio,
                        audio,
                        step_name,
                        base_path
                    )
            
            return step_analysis
            
        except Exception as e:
            logger.error(f"Step analysis failed for {step_name}: {str(e)}")
            return {}

    def _visualize_comparison(
        self,
        prev_audio: torch.Tensor,
        curr_audio: torch.Tensor,
        step_name: str,
        base_path: str
    ) -> None:
        """Create comparative visualizations between processing steps"""
        try:
            # Waveform comparison
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            
            ax1.plot(prev_audio.squeeze().numpy())
            ax1.set_title("Before Processing")
            ax1.grid(True)
            
            ax2.plot(curr_audio.squeeze().numpy())
            ax2.set_title(f"After {step_name}")
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{base_path}_waveform_comparison.png")
            plt.close()
            
            # Spectrogram comparison
            # Spectrogram comparison
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            for ax, audio, title in [
                (ax1, prev_audio, "Before Processing"),
                (ax2, curr_audio, f"After {step_name}")
            ]:
                spec = torch.stft(
                    audio.squeeze(),
                    n_fft=2048,
                    hop_length=512,
                    win_length=2048,
                    window=torch.hann_window(2048).to(audio.device),
                    return_complex=True
                )
                spec_mag = torch.abs(spec).numpy()
                
                im = ax.imshow(
                    20 * np.log10(spec_mag + 1e-10),
                    aspect='auto',
                    origin='lower',
                    extent=[0, len(audio.squeeze())/16000, 0, 8000]
                )
                ax.set_title(title)
                fig.colorbar(im, ax=ax, label='dB')
            
            plt.tight_layout()
            plt.savefig(f"{base_path}_spectrogram_comparison.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Comparison visualization failed: {str(e)}")

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
        
        # Add spectral analysis if available
        if 'spectral_analysis' in analysis:
            report.append("\nSpectral Analysis:")
            report.append("-" * 20)
            for band, energy in analysis['spectral_analysis']['frequency_bands'].items():
                report.append(f"- {band.title()} Band Energy: {energy:.3f}")
            report.append(f"- Balance Score: {analysis['spectral_analysis']['spectral_metrics']['band_balance_score']:.3f}")
        
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

    def generate_step_report(self) -> str:
        """Generate a comprehensive report of all processing steps"""
        report = ["Processing Step Analysis", "=" * 30]
        
        for step_name, analysis in self.step_diagnostics.items():
            report.extend([
                f"\n{step_name}:",
                "-" * len(step_name),
                "\nWaveform Metrics:",
                f"  Peak Amplitude: {analysis['waveform'].get('peak_amplitude', 0):.3f}",
                f"  RMS Level: {analysis['waveform'].get('rms_level', 0):.3f}",
                f"  Silence %: {analysis['waveform'].get('silence_percentage', 0):.1f}%",
                "\nSpectral Balance:",
                f"  Balance Score: {analysis['spectral'].get('band_balance_score', 0):.3f}",
                f"  Speech Intelligibility: {self.spectral_metrics.get('speech_intelligibility', 0):.3f}",
                f"  Vocal Warmth: {self.spectral_metrics.get('vocal_warmth', 0):.3f}",
                f"  Clarity: {self.spectral_metrics.get('clarity', 0):.3f}"
            ])
        
        return "\n".join(report)