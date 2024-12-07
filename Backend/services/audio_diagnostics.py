import logging
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import json
from scipy import signal
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AudioQualityLevel(Enum):
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class FrequencyBand:
    name: str
    low_freq: float
    high_freq: float
    description: str
    perceptual_features: List[str]

class AudioDiagnostics:
    """
    Advanced diagnostic tool for comprehensive audio quality analysis with special focus
    on speech synthesis and translation quality assessment.
    """

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
                'clipping_points': int(np.sum(np.abs(audio_np) > 0.99))
            }
        
            return analysis
            
        except Exception as e:
            logger.error(f"Waveform analysis failed: {str(e)}")
            return {}
    
    # Frequency bands with perceptual mappings
    FREQUENCY_BANDS = {
        'sub_bass': FrequencyBand(
            'Sub-bass', 20, 60,
            "Foundation frequencies, rarely crucial for speech",
            ['warmth', 'rumble']
        ),
        'bass': FrequencyBand(
            'Bass', 60, 250,
            "Fundamental speech frequencies, important for vowels",
            ['fullness', 'resonance', 'voice foundation']
        ),
        'low_mids': FrequencyBand(
            'Low-mids', 250, 500,
            "Critical for speech intelligibility and vowel clarity",
            ['vowel clarity', 'speech body', 'warmth']
        ),
        'mids': FrequencyBand(
            'Mids', 500, 2000,
            "Core speech frequencies, essential for intelligibility",
            ['intelligibility', 'presence', 'clarity']
        ),
        'upper_mids': FrequencyBand(
            'Upper-mids', 2000, 4000,
            "Consonant definition and speech clarity",
            ['definition', 'articulation', 'consonant clarity']
        ),
        'presence': FrequencyBand(
            'Presence', 4000, 6000,
            "Sibilance and speech brightness",
            ['sibilance', 'air', 'closeness']
        ),
        'brilliance': FrequencyBand(
            'Brilliance', 6000, 20000,
            "Air and spaciousness in speech",
            ['air', 'sparkle', 'space']
        )
    }

    def __init__(self):
        """Initialize the enhanced audio diagnostics system"""
        # Create diagnostics directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diagnostics_dir = Path(f"diagnostics/{self.timestamp}")
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
        # Basic quality metrics (required for core functionality)
        self.quality_metrics = {
            'robotic_voice': 0,      # 1-5 scale
            'pronunciation': 0,       # 1-5 scale
            'audio_clarity': 0,      # 1-5 scale
            'background_noise': 0,    # 1-5 scale
            'voice_consistency': 0,   # 1-5 scale
            'spectral_balance': 0     # 1-5 scale
        }

        # Spectral metrics (required for core functionality)
        self.spectral_metrics = {
            'low_band_energy': 0.0,
            'mid_band_energy': 0.0,
            'high_band_energy': 0.0,
            'band_balance_score': 0.0
        }

        # Common issues (required for core functionality)
        self.common_issues = {
            'clipping': False,
            'metallic_artifacts': False,
            'sibilance': False,
            'choppy': False,
            'muffled': False,
            'echo': False,
            'spectral_imbalance': False
        }

        # Language-specific issues (required for core functionality)
        self.language_specific_issues = {}
    
        # Initialize comprehensive quality metrics
        self.acoustic_metrics = {
            'technical': {
                'signal_to_noise_ratio': 0.0,
                'harmonic_to_noise_ratio': 0.0,
                'peak_to_rms_ratio': 0.0,
                'dc_offset': 0.0,
                'phase_correlation': 0.0,
                'spectral_flatness': 0.0,
                'spectral_centroid': 0.0,
                'spectral_bandwidth': 0.0,
                'spectral_rolloff': 0.0
            },
            'perceptual': {
                'clarity_index': 0.0,
                'definition_index': 0.0,
                'reverb_time': 0.0,
                'early_decay_time': 0.0,
                'speech_transmission_index': 0.0
            },
            'speech_specific': {
                'articulation_index': 0.0,
                'formant_clarity': 0.0,
                'pitch_stability': 0.0,
                'voice_consistency': 0.0,
                'pronunciation_accuracy': 0.0
            }
        }
    
        self.frequency_analysis = {band: {
            'energy': 0.0,
            'peak_frequency': 0.0,
            'variance': 0.0,
            'coherence': 0.0,
            'perceptual_impact': 0.0
        } for band in self.FREQUENCY_BANDS.keys()}
    
        # Dynamic range analysis
        self.dynamic_metrics = {
            'peak_level': 0.0,
            'rms_level': 0.0,
            'dynamic_range': 0.0,
            'loudness_range': 0.0,
            'true_peak': 0.0,
            'integrated_loudness': 0.0,
            'loudness_consistency': 0.0
        }
    
        # Initialize temporal analysis storage
        self.temporal_analysis = {
            'envelope': {
                'attack_time': 0.0,
                'decay_time': 0.0,
                'sustain_level': 0.0,
                'release_time': 0.0
            },
            'modulation': {
                'rate': 0.0,
                'depth': 0.0,
                'periodicity': 0.0
            },
            'rhythm': {
                'tempo': 0.0,
                'regularity': 0.0,
                'speech_rate': 0.0
            }
        }
    
        # Speech quality metrics
        self.speech_quality = {
            'intelligibility': {
                'score': 0.0,
                'confidence': 0.0,
                'problem_areas': []
            },
            'naturalness': {
                'score': 0.0,
                'artifacts': [],
                'deviations': []
            },
            'prosody': {
                'pitch_range': 0.0,
                'stress_accuracy': 0.0,
                'rhythm_score': 0.0,
                'intonation_accuracy': 0.0
            }
        }
    
        # Processing artifacts tracking
        self.artifacts = {
            'clipping': {
                'detected': False,
                'count': 0,
                'severity': 0.0,
                'timestamps': []
            },
            'distortion': {
                'detected': False,
                'type': None,
                'severity': 0.0,
                'frequency_ranges': []
            },
            'dropout': {
                'detected': False,
                'count': 0,
                'durations': [],
                'timestamps': []
            },
            'noise': {
                'detected': False,
                'type': None,
                'level': 0.0,
                'spectrum': None
            }
        }
    
        # Initialize analysis tools
        self._init_analysis_tools()
    
    def analyze_spectral_balance(self, audio: torch.Tensor) -> Dict[str, float]:
        """
        Analyze spectral balance of the audio by computing energy in different frequency bands.

        Args:
            audio (torch.Tensor): Input audio tensor (mono or stereo)
            
        Returns:
            Dict[str, float]: Energy in low, mid, and high frequency bands
        """
        try:
            # Input validation
            if not isinstance(audio, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")

            if audio.dim() == 0 or audio.size(0) == 0:
                raise ValueError("Empty audio tensor")

            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
                
            # Compute STFT with error checking
            try:
                spec = torch.stft(
                    audio.squeeze(),
                    n_fft=4096,
                    hop_length=1024,
                    win_length=4096,
                    window=torch.hann_window(4096).to(audio.device),
                    return_complex=True
                )
            except RuntimeError as e:
               logger.error(f"STFT computation failed: {str(e)}")
               raise
            
            mag = torch.abs(spec)
            
            # Validate magnitude spectrum
            if torch.isnan(mag).any() or torch.isinf(mag).any():
                raise ValueError("Invalid magnitude spectrum")
        
            # Calculate energy in different frequency bands
            freq_bands = {
                'low': float(torch.mean(mag[:, :mag.shape[1]//4]).item()),
                'mid': float(torch.mean(mag[:, mag.shape[1]//4:mag.shape[1]//2]).item()),
                'high': float(torch.mean(mag[:, mag.shape[1]//2:]).item())
            }
            
            # Validate results
            if any(not isinstance(v, float) or np.isnan(v) or np.isinf(v) for v in freq_bands.values()):
                raise ValueError("Invalid frequency band values computed")
        
            return freq_bands
            
        except Exception as e:
            logger.error(f"Spectral balance analysis failed: {str(e)}")
            return {'low': 0.0, 'mid': 0.0, 'high': 0.0}
    
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
            
            # Input validation
            if not isinstance(analysis, dict):
                raise ValueError("Invalid analysis data")
            
            if not analysis:
                raise ValueError("Empty analysis dictionary")

            if not isinstance(target_language, str) or not target_language:
                raise ValueError("Invalid target language")

            report = [
                "Audio Quality Analysis Report",
                "=" * 30,
                f"\nTarget Language: {target_language}",
                "\nWaveform Analysis:",
                "-" * 20
            ]
        
            # Add waveform metrics
            if 'waveform_analysis' in analysis:
                for metric, value in analysis['waveform_analysis'].items():
                    try:
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

                spectral_metrics = analysis['spectral_analysis'].get('spectral_metrics', {})
                if 'band_balance_score' in spectral_metrics:
                    try:
                        report.append(f"- Balance Score: {float(spectral_metrics['band_balance_score']):.3f}")
                    except (TypeError, ValueError) as e:
                        report.append("- Balance Score: N/A")
            
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
            logger.debug(f"Analysis data: {analysis}")  # Debug log to see what's coming in

            
            if 'language_specific' in analysis and analysis['language_specific']:
                report.append("\nLanguage-Specific Issues:")
                report.append("-" * 20)
                detected = False
                for issue, present in analysis['language_specific'].items():
                    if present:
                        report.append(f"- {issue.replace('_', ' ').title()}")
                        detected = True
                if not detected:
                    report.append("No language-specific issues detected")
            
            return "\n".join(report)
        
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return f"Error generating report: {str(e)}"

    def _init_analysis_tools(self):
        """Initialize advanced DSP tools and configurations"""
        self.analysis_config = {
            'stft': {
                'window_sizes': [512, 1024, 2048, 4096],
                'hop_ratios': [0.25, 0.5, 0.75],
                'window_types': ['hann', 'hamming', 'blackman']
            },
            'filter_banks': {
                'mel': {'n_mels': 128, 'fmin': 20, 'fmax': 8000},
                'bark': {'n_bands': 24},
                'erb': {'n_bands': 32}
            },
            'thresholds': {
                'clipping': 0.99,
                'noise_floor': -60,
                'silence': -40,
                'sibilance': 0.7
            }
        }
    
    def analyze_spectral_characteristics(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive spectral analysis providing detailed frequency-domain insights.
        """
        try:
            # Multi-resolution spectral analysis
            spectral_data = {}
            for window_size in self.analysis_config['stft']['window_sizes']:
                spec = torch.stft(
                    audio.squeeze(),
                    n_fft=window_size,
                    hop_length=int(window_size * 0.25),
                    win_length=window_size,
                    window=torch.hann_window(window_size).to(audio.device),
                    return_complex=True
                )
                mag = torch.abs(spec)
                phase = torch.angle(spec)
                
                # Detailed frequency band analysis
                band_data = {}
                for band_name, band_info in self.FREQUENCY_BANDS.items():
                    # Convert frequencies to bin indices
                    low_bin = int(band_info.low_freq * window_size / 16000)
                    high_bin = int(band_info.high_freq * window_size / 16000)
                    
                    # Band-specific analysis
                    band_content = mag[:, low_bin:high_bin]
                    band_data[band_name] = {
                        'mean_energy': float(torch.mean(band_content).item()),
                        'peak_energy': float(torch.max(band_content).item()),
                        'energy_variation': float(torch.std(band_content).item()),
                        'spectral_slope': self._calculate_spectral_slope(band_content),
                        'temporal_evolution': self._analyze_temporal_evolution(band_content),
                        'perceptual_description': self._describe_band_characteristics(
                            band_content, band_info.perceptual_features
                        )
                    }
                
                spectral_data[f'resolution_{window_size}'] = {
                    'band_analysis': band_data,
                    'overall_characteristics': {
                        'spectral_centroid': self._calculate_spectral_centroid(mag),
                        'spectral_spread': self._calculate_spectral_spread(mag),
                        'spectral_skewness': self._calculate_spectral_skewness(mag),
                        'spectral_kurtosis': self._calculate_spectral_kurtosis(mag),
                        'spectral_flatness': self._calculate_spectral_flatness(mag),
                        'spectral_rolloff': self._calculate_spectral_rolloff(mag)
                    }
                }
            
            return spectral_data
            
        except Exception as e:
            logger.error(f"Spectral analysis failed: {str(e)}")
            return {}

    def analyze_temporal_structure(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze temporal characteristics of the audio with focus on speech-relevant features.
        """
        try:
            audio_np = audio.squeeze().numpy()
            
            # Envelope analysis
            analytic_signal = signal.hilbert(audio_np)
            envelope = np.abs(analytic_signal)
            
            # Detect speech segments and pauses
            is_speech = envelope > np.mean(envelope) * 0.1
            speech_segments = self._find_continuous_segments(is_speech)
            
            # Temporal pattern analysis
            temporal_data = {
                'speech_segments': {
                    'count': len(speech_segments),
                    'mean_duration': float(np.mean([s[1] - s[0] for s in speech_segments])),
                    'duration_variance': float(np.var([s[1] - s[0] for s in speech_segments])),
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

    def _describe_temporal_patterns(self, segments: List[Tuple[int, int]]) -> str:
        """Generate natural language description of temporal patterns."""
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

    def analyze_perceptual_quality(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze perceptual aspects of speech quality with detailed descriptions.
        """
        try:
            # Convert to numpy for processing
            audio_np = audio.squeeze().numpy()
            
            # Speech quality metrics
            quality_metrics = {
                'clarity': self._assess_clarity(audio_np),
                'naturalness': self._assess_naturalness(audio_np),
                'articulation': self._assess_articulation(audio_np),
                'voice_quality': self._analyze_voice_characteristics(audio_np),
                'emotional_characteristics': self._analyze_emotional_content(audio_np)
            }
            
            # Generate detailed descriptions
            perceptual_analysis = {
                'metrics': quality_metrics,
                'detailed_description': self._generate_perceptual_description(quality_metrics),
                'quality_issues': self._identify_perceptual_issues(quality_metrics),
                'suggested_improvements': self._suggest_quality_improvements(quality_metrics)
            }
            
            return perceptual_analysis
            
        except Exception as e:
            logger.error(f"Perceptual analysis failed: {str(e)}")
            return {}

    def _analyze_voice_characteristics(self, audio_np: np.ndarray) -> Dict[str, Any]:
        """Analyze detailed voice characteristics."""
        characteristics = {
            'timbre': {
                'brightness': self._calculate_brightness(audio_np),
                'warmth': self._calculate_warmth(audio_np),
                'resonance': self._calculate_resonance(audio_np),
                'description': self._describe_timbre(audio_np)
            },
            'consistency': {
                'pitch_stability': self._analyze_pitch_stability(audio_np),
                'volume_stability': self._analyze_volume_stability(audio_np),
                'quality_stability': self._analyze_quality_stability(audio_np)
            },
            'naturalness_indicators': {
                'micro_variations': self._analyze_micro_variations(audio_np),
                'formant_structure': self._analyze_formant_structure(audio_np),
                'breathiness': self._calculate_breathiness(audio_np)
            }
        }
        
        return characteristics

    def _describe_timbre(self, audio_np: np.ndarray) -> str:
        """Generate detailed description of voice timbre."""
        brightness = self._calculate_brightness(audio_np)
        warmth = self._calculate_warmth(audio_np)
        
        descriptions = []
        if brightness > 0.7:
            descriptions.append("The voice exhibits a bright, present quality with pronounced upper harmonics")
        elif brightness < 0.3:
            descriptions.append("The voice has a darker, more subdued tonal quality")
            
        if warmth > 0.7:
            descriptions.append("Rich, warm resonance in the lower-mid frequencies")
        elif warmth < 0.3:
            descriptions.append("Lean, clinical tonal character with limited warmth")
            
        resonance = self._calculate_resonance(audio_np)
        if resonance > 0.7:
            descriptions.append("Strong resonant qualities suggesting good vocal projection")
        elif resonance < 0.3:
            descriptions.append("Limited resonance indicating potential lack of depth or body in the voice")
            
        return ". ".join(descriptions)
    
    def analyze_language_specific_features(self, audio: torch.Tensor, target_language: str) -> Dict[str, Any]:
        """
        Comprehensive language-specific analysis with focus on phonetic and prosodic features.
        """
        language_features = {
            'fra': {
                'nasalization': self._analyze_french_nasalization(audio),
                'liaison': self._analyze_french_liaison(audio),
                'prosody': self._analyze_french_prosody(audio),
                'vowel_space': {
                    'front_rounded': self._analyze_vowel_space(audio, 'front_rounded'),
                    'front_unrounded': self._analyze_vowel_space(audio, 'front_unrounded'),
                    'back_rounded': self._analyze_vowel_space(audio, 'back_rounded'),
                    'nasal_vowels': self._analyze_vowel_space(audio, 'nasal')
                },
                'rhythm_patterns': self._analyze_french_rhythm(audio),
                'intonation': self._analyze_french_intonation(audio),
                'consonant_features': {
                    'uvular_r': self._analyze_french_r(audio),
                    'final_consonants': self._analyze_final_consonants(audio)
                }
            },
            'deu': {
                'vowel_length': self._analyze_german_vowel_length(audio),
                'umlauts': self._analyze_german_umlauts(audio),
                'consonant_clusters': self._analyze_german_consonants(audio),
                'word_stress': self._analyze_german_stress(audio),
                'glottal_stops': self._analyze_glottal_stops(audio),
                'final_devoicing': self._analyze_final_devoicing(audio),
                'schwa_realization': self._analyze_schwa(audio)
            },
            'spa': {
                'phoneme_analysis': {
                    'trilled_r': self._analyze_spanish_trill(audio),
                    'interdental_theta': self._analyze_interdental(audio),
                    'stop_consonants': self._analyze_spanish_stops(audio)
                },
                'syllable_timing': self._analyze_spanish_timing(audio),
                'intonation_patterns': self._analyze_spanish_intonation(audio),
                'vowel_clarity': self._analyze_spanish_vowels(audio),
                'stress_patterns': self._analyze_spanish_stress(audio)
            },
            'ita': {
                'gemination': self._analyze_italian_gemination(audio),
                'vowel_quality': self._analyze_italian_vowels(audio),
                'consonant_precision': self._analyze_italian_consonants(audio),
                'prosodic_features': {
                    'stress_timing': self._analyze_italian_stress(audio),
                    'intonation': self._analyze_italian_intonation(audio),
                    'rhythm': self._analyze_italian_rhythm(audio)
                }
            },
            'por': {
                'vowel_analysis': {
                    'nasal_vowels': self._analyze_portuguese_nasalization(audio),
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
        }

        if target_language not in language_features:
            logger.warning(f"No specific analysis available for language: {target_language}")
            return {}

        try:
            analysis = language_features[target_language]
            
            # Add linguistic context
            analysis['linguistic_context'] = self._generate_linguistic_context(target_language)
            
            # Generate comprehensive description
            analysis['detailed_assessment'] = self._generate_language_assessment(
                analysis, target_language
            )
            
            return analysis
        except Exception as e:
            logger.error(f"Language-specific analysis failed: {str(e)}")
            return {}

    def _analyze_french_nasalization(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze French nasal vowel characteristics."""
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
            
            # Analysis of nasal resonances (500-2000 Hz)
            nasal_band = mag[:, 25:100]  # Approximate band for nasal resonances
            
            analysis = {
                'nasal_resonance': {
                    'strength': float(torch.mean(nasal_band).item()),
                    'stability': float(torch.std(nasal_band).item()),
                    'peak_frequencies': self._find_peak_frequencies(nasal_band),
                },
                'formant_structure': self._analyze_nasal_formants(mag),
                'temporal_evolution': self._analyze_temporal_stability(nasal_band),
                'quality_assessment': {
                    'authenticity': self._assess_nasal_authenticity(nasal_band),
                    'consistency': self._assess_nasal_consistency(nasal_band),
                    'distinction': self._assess_nasal_distinction(mag)
                }
            }
            
            # Generate descriptive assessment
            analysis['description'] = self._describe_nasal_qualities(analysis)
            
            return analysis
        except Exception as e:
            logger.error(f"French nasalization analysis failed: {str(e)}")
            return {}

    def _describe_nasal_qualities(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed description of nasal vowel qualities."""
        descriptions = []
        
        # Assess resonance strength
        strength = analysis['nasal_resonance']['strength']
        if strength > 0.7:
            descriptions.append("Strong and prominent nasal resonances, characteristic of authentic French nasal vowels")
        elif strength > 0.4:
            descriptions.append("Moderate nasal resonance, suggesting adequate but not optimal nasalization")
        else:
            descriptions.append("Weak nasal resonance, indicating potential issues with nasal vowel production")

        # Assess stability
        stability = analysis['nasal_resonance']['stability']
        if stability < 0.2:
            descriptions.append("Highly consistent nasal resonance throughout vowel production")
        elif stability < 0.4:
            descriptions.append("Moderately stable nasal resonance with some variation")
        else:
            descriptions.append("Unstable nasal resonance, suggesting inconsistent nasalization")

        # Assess distinction
        if analysis['quality_assessment']['distinction'] > 0.7:
            descriptions.append("Clear distinction between nasal and oral vowels")
        else:
            descriptions.append("Limited distinction between nasal and oral vowels, potentially affecting comprehensibility")

        return ". ".join(descriptions)

    def generate_comprehensive_report(self, 
                                   audio: torch.Tensor, 
                                   target_language: str,
                                   include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report with detailed descriptions and optional visualizations.
        """
        try:
            # Perform all analyses
            spectral_analysis = self.analyze_spectral_characteristics(audio)
            temporal_analysis = self.analyze_temporal_structure(audio)
            perceptual_analysis = self.analyze_perceptual_quality(audio)
            language_analysis = self.analyze_language_specific_features(audio, target_language)
            
            # Generate summary scores
            quality_scores = {
                'technical_quality': self._calculate_technical_score(spectral_analysis),
                'perceptual_quality': self._calculate_perceptual_score(perceptual_analysis),
                'linguistic_accuracy': self._calculate_linguistic_score(language_analysis),
                'overall_quality': self._calculate_overall_score(
                    spectral_analysis,
                    temporal_analysis,
                    perceptual_analysis,
                    language_analysis
                )
            }
            
            # Generate natural language descriptions
            descriptions = {
                'technical_description': self._describe_technical_quality(spectral_analysis, temporal_analysis),
                'perceptual_description': self._describe_perceptual_quality(perceptual_analysis),
                'linguistic_description': self._describe_linguistic_quality(language_analysis, target_language),
                'overall_assessment': self._generate_overall_assessment(quality_scores)
            }
            
            # Compile report
            report = {
                'summary': {
                    'quality_scores': quality_scores,
                    'key_findings': self._identify_key_findings(
                        spectral_analysis,
                        temporal_analysis,
                        perceptual_analysis,
                        language_analysis
                    ),
                    'recommendations': self._generate_recommendations(quality_scores)
                },
                'detailed_analysis': {
                    'spectral_characteristics': spectral_analysis,
                    'temporal_structure': temporal_analysis,
                    'perceptual_quality': perceptual_analysis,
                    'language_specific_features': language_analysis
                },
                'descriptions': descriptions,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'target_language': target_language,
                    'audio_duration': audio.shape[-1] / 16000,  # Assuming 16kHz sample rate
                    'analysis_version': '2.0.0'
                }
            }
            
            if include_visualizations:
                report['visualizations'] = self._generate_visualizations(
                    audio,
                    spectral_analysis,
                    temporal_analysis
                )
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_french_liaison(self, audio: torch.Tensor) -> bool:
        """Analyze French liaison patterns."""
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
            liaison_score = self._analyze_temporal_stability(mag)
            return liaison_score < 0.6
        except Exception as e:
            logger.error(f"French liaison analysis failed: {str(e)}")
            return False

    def _check_french_prosody(self, audio: torch.Tensor) -> bool:
        """Check French prosodic patterns."""
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
            energy_contour = torch.mean(mag, dim=1)
            prosody_score = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(prosody_score) < 0.5
        
        except Exception as e:
            logger.error(f"French prosody check failed: {str(e)}")
            return False

    def _check_french_vowels(self, audio: torch.Tensor) -> bool:
        """Check French vowel quality."""
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
            vowel_region = mag[:, :mag.shape[1]//2]
            vowel_score = torch.mean(vowel_region) / torch.max(vowel_region)
            return float(vowel_score) < 0.6
        
        except Exception as e:
            logger.error(f"French vowel check failed: {str(e)}")
            return False
        
    def _analyze_german_vowel_length(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze German vowel length and formant structure."""
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
        
            analysis = {
                'formant_structure': {
                    'accuracy': float(torch.mean(formant_region).item()),
                    'stability': float(torch.std(formant_region).item()),
                },
                'temporal_evolution': {
                    'precision': self._analyze_temporal_stability(formant_region),
                },
                'prosody': {
                    'accuracy': self._analyze_german_prosody(mag),
                },
                'consonant_features': {
                    'precision': self._analyze_german_consonant_precision(mag),
                }
            }
        
            return analysis
        except Exception as e:
            logger.error(f"German vowel analysis failed: {str(e)}")
            return {
                'formant_structure': {'accuracy': 0.0, 'stability': 0.0},
                'temporal_evolution': {'precision': 0.0},
                'prosody': {'accuracy': 0.0},
                'consonant_features': {'precision': 0.0}
            }

    def _analyze_german_prosody(self, mag: torch.Tensor) -> float:
        """Analyze German prosodic patterns."""
        try:
            # Analyze energy distribution for stress patterns
            energy_contour = torch.mean(mag, dim=1)
            stress_variation = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(min(1.0, stress_variation))
        except Exception as e:
            logger.error(f"German prosody analysis failed: {str(e)}")
            return 0.0

    def _analyze_german_consonant_precision(self, mag: torch.Tensor) -> float:
        """Analyze precision of German consonant articulation."""
        try:
            # Focus on high-frequency content for consonants
            consonant_region = mag[:, mag.shape[1]//2:]
            precision = torch.mean(consonant_region) / torch.max(consonant_region)
            return float(min(1.0, precision))
        except Exception as e:
            logger.error(f"German consonant analysis failed: {str(e)}")
            return 0.0

    def _analyze_italian_gemination(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Italian gemination and related features."""
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
        
            analysis = {
                'consonant_features': {
                    'precision': self._analyze_italian_consonants(mag),
                },
                'vowel_features': {
                    'accuracy': self._analyze_italian_vowels(mag),
                },
                'articulation': {
                    'clarity': self._analyze_italian_articulation(mag),
                },
                'prosody': {
                    'rhythm': self._analyze_italian_rhythm(mag),
                }
            }
        
            return analysis
        except Exception as e:
            logger.error(f"Italian gemination analysis failed: {str(e)}")
            return {
                'consonant_features': {'precision': 0.0},
                'vowel_features': {'accuracy': 0.0},
                'articulation': {'clarity': 0.0},
                'prosody': {'rhythm': 0.0}
            }

    def _analyze_italian_consonants(self, mag: torch.Tensor) -> float:
        """Analyze Italian consonant characteristics."""
        try:
            consonant_region = mag[:, mag.shape[1]//2:]
            precision = torch.mean(consonant_region) / torch.max(consonant_region)
            return float(min(1.0, precision))
        except Exception as e:
            logger.error(f"Italian consonant analysis failed: {str(e)}")
            return 0.0

    def _analyze_italian_vowels(self, mag: torch.Tensor) -> float:
        """Analyze Italian vowel characteristics."""
        try:
            vowel_region = mag[:, :mag.shape[1]//2]
            accuracy = torch.mean(vowel_region) / torch.max(vowel_region)
            return float(min(1.0, accuracy))
        except Exception as e:
            logger.error(f"Italian vowel analysis failed: {str(e)}")
            return 0.0

    def _analyze_italian_articulation(self, mag: torch.Tensor) -> float:
        """Analyze clarity of Italian articulation."""
        try:
            clarity = torch.std(torch.mean(mag, dim=1)) / torch.mean(mag)
            return float(min(1.0, clarity))
        except Exception as e:
            logger.error(f"Italian articulation analysis failed: {str(e)}")
            return 0.0

    def _analyze_italian_rhythm(self, mag: torch.Tensor) -> float:
        """Analyze Italian rhythmic patterns."""
        try:
            energy_contour = torch.mean(mag, dim=1)
            rhythm_score = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(min(1.0, rhythm_score))
        except Exception as e:
            logger.error(f"Italian rhythm analysis failed: {str(e)}")
            return 0.0
        
    def _analyze_portuguese_nasalization(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Portuguese nasalization and related features."""
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
        
            analysis = {
                'vowel_features': {
                    'authenticity': self._analyze_portuguese_vowels(mag),
                    'reduction': self._analyze_portuguese_reduction(mag),
                },
                'prosody': {
                    'accuracy': self._analyze_portuguese_prosody(mag),
                },
                'consonant_features': {
                    'precision': self._analyze_portuguese_consonants(mag),
                }
            }
        
            return analysis
        except Exception as e:
            logger.error(f"Portuguese nasalization analysis failed: {str(e)}")
            return {
                'vowel_features': {'authenticity': 0.0, 'reduction': 0.0},
                'prosody': {'accuracy': 0.0},
                'consonant_features': {'precision': 0.0}
            }

    def _analyze_portuguese_vowels(self, mag: torch.Tensor) -> float:
        """Analyze Portuguese vowel characteristics."""
        try:
            vowel_region = mag[:, :mag.shape[1]//2]
            authenticity = torch.mean(vowel_region) / torch.max(vowel_region)
            return float(min(1.0, authenticity))
        except Exception as e:
            logger.error(f"Portuguese vowel analysis failed: {str(e)}")
            return 0.0

    def _analyze_portuguese_reduction(self, mag: torch.Tensor) -> float:
        """Analyze Portuguese vowel reduction patterns."""
        try:
            energy_contour = torch.mean(mag, dim=1)
            reduction = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(min(1.0, reduction))
        except Exception as e:
            logger.error(f"Portuguese reduction analysis failed: {str(e)}")
            return 0.0

    def _analyze_portuguese_prosody(self, mag: torch.Tensor) -> float:
        """Analyze Portuguese prosodic patterns."""
        try:
            energy_contour = torch.mean(mag, dim=1)
            accuracy = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(min(1.0, accuracy))
        except Exception as e:
            logger.error(f"Portuguese prosody analysis failed: {str(e)}")
            return 0.0

    def _analyze_portuguese_consonants(self, mag: torch.Tensor) -> float:
        """Analyze Portuguese consonant characteristics."""
        try:
            consonant_region = mag[:, mag.shape[1]//2:]
            precision = torch.mean(consonant_region) / torch.max(consonant_region)
            return float(min(1.0, precision))
        except Exception as e:
            logger.error(f"Portuguese consonant analysis failed: {str(e)}")
            return 0.0

    def _analyze_spanish_trill(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze Spanish trill and related features."""
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
        
            analysis = {
                'consonant_features': {
                    'precision': self._analyze_spanish_consonants(mag),
                },
                'vowel_features': {
                    'clarity': self._analyze_spanish_vowels(mag),
                },
                'prosody': {
                    'accuracy': self._analyze_spanish_prosody(mag),
                },
                'articulation': {
                    'precision': self._analyze_spanish_articulation(mag),
                }
            }
        
            return analysis
        except Exception as e:
            logger.error(f"Spanish trill analysis failed: {str(e)}")
            return {
                'consonant_features': {'precision': 0.0},
                'vowel_features': {'clarity': 0.0},
                'prosody': {'accuracy': 0.0},
                'articulation': {'precision': 0.0}
            }

    def _analyze_spanish_consonants(self, mag: torch.Tensor) -> float:
        """Analyze Spanish consonant characteristics."""
        try:
            consonant_region = mag[:, mag.shape[1]//2:]
            precision = torch.mean(consonant_region) / torch.max(consonant_region)
            return float(min(1.0, precision))
        except Exception as e:
            logger.error(f"Spanish consonant analysis failed: {str(e)}")
            return 0.0

    def _analyze_spanish_vowels(self, mag: torch.Tensor) -> float:
        """Analyze Spanish vowel characteristics."""
        try:
            vowel_region = mag[:, :mag.shape[1]//2]
            clarity = torch.mean(vowel_region) / torch.max(vowel_region)
            return float(min(1.0, clarity))
        except Exception as e:
            logger.error(f"Spanish vowel analysis failed: {str(e)}")
            return 0.0

    def _analyze_spanish_prosody(self, mag: torch.Tensor) -> float:
        """Analyze Spanish prosodic patterns."""
        try:
            energy_contour = torch.mean(mag, dim=1)
            accuracy = torch.std(energy_contour) / torch.mean(energy_contour)
            return float(min(1.0, accuracy))
        except Exception as e:
            logger.error(f"Spanish prosody analysis failed: {str(e)}")
            return 0.0

    def _analyze_spanish_articulation(self, mag: torch.Tensor) -> float:
        """Analyze Spanish articulation precision."""
        try:
            precision = torch.std(torch.mean(mag, dim=1)) / torch.mean(mag)
            return float(min(1.0, precision))
        except Exception as e:
            logger.error(f"Spanish articulation analysis failed: {str(e)}")
            return 0.0
    
    def _analyze_language_specific(self, audio: torch.Tensor, target_language: str, spectral_bands: Dict[str, float]) -> None:
        """Analyze language-specific characteristics and update language_specific_issues."""
        try:
        
            # French analysis
            if target_language == 'fra':
                nasal_analysis = self._analyze_french_nasalization(audio)
                self.language_specific_issues['fra'] = {
                    'nasalization': nasal_analysis['quality_assessment']['authenticity'] < 0.6,
                    'liaison': self._analyze_french_liaison(audio),
                    'prosody': self._check_french_prosody(audio),
                    'vowel_quality': self._check_french_vowels(audio)
                }
                
            # German analysis
            elif target_language == 'deu':
                vowel_analysis = self._analyze_german_vowel_length(audio)
                self.language_specific_issues['deu'] = {
                    'umlaut_issues': vowel_analysis['formant_structure']['accuracy'] < 0.6,
                    'consonant_clusters': vowel_analysis['temporal_evolution']['precision'] < 0.5,
                    'word_stress': vowel_analysis['prosody']['accuracy'] < 0.6,
                    'final_devoicing': vowel_analysis['consonant_features']['precision'] < 0.5
                }
                
            # Italian analysis
            elif target_language == 'ita':
                italian_analysis = self._analyze_italian_gemination(audio)
                self.language_specific_issues['ita'] = {
                    'gemination': italian_analysis['consonant_features']['precision'] < 0.6,
                    'vowel_length': italian_analysis['vowel_features']['accuracy'] < 0.6,
                    'consonant_precision': italian_analysis['articulation']['clarity'] < 0.5,
                    'stress_timing': italian_analysis['prosody']['rhythm'] < 0.6
                }
                
            # Spanish analysis
            elif target_language == 'spa':
                spanish_analysis = self._analyze_spanish_trill(audio)
                self.language_specific_issues['spa'] = {
                    'trill': spanish_analysis['consonant_features']['precision'] < 0.6,
                    'vowel_clarity': spanish_analysis['vowel_features']['clarity'] < 0.6,
                    'stress_patterns': spanish_analysis['prosody']['accuracy'] < 0.5,
                    'consonant_precision': spanish_analysis['articulation']['precision'] < 0.6
                }
            
            # Portuguese analysis
            elif target_language == 'por':
                portuguese_analysis = self._analyze_portuguese_nasalization(audio)
                self.language_specific_issues['por'] = {
                    'nasalization': portuguese_analysis['vowel_features']['authenticity'] < 0.6,
                    'vowel_reduction': portuguese_analysis['vowel_features']['reduction'] < 0.5,
                    'stress_patterns': portuguese_analysis['prosody']['accuracy'] < 0.6,
                    'consonant_palatalization': portuguese_analysis['consonant_features']['precision'] < 0.5
                }
        except Exception as e:
            logger.error(f"Language-specific analysis failed for {target_language}: {str(e)}")
    
    
    def analyze_translation(self, audio: torch.Tensor, target_language: str) -> dict:
        """
        Analyze translation output comprehensively.
        
        Args:
            audio: Audio tensor to analyze
            target_language: Target language code (e.g., 'fra', 'deu', etc.)
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            # Basic waveform analysis
            waveform_analysis = self.analyze_waveform(audio)
            
            # Spectral analysis
            spectral_bands = self.analyze_spectral_balance(audio)
            
            # Language-specific analysis
            if target_language in self.language_specific_issues:
                self._analyze_language_specific(audio, target_language, spectral_bands)
            
            # Calculate quality metrics
            quality_metrics = {
                'robotic_voice': self._calculate_robotic_score(audio),
                'pronunciation': self._calculate_pronunciation_score(audio),
                'audio_clarity': self._calculate_clarity_score(audio),
                'background_noise': self._calculate_noise_score(audio),
                'voice_consistency': self._calculate_consistency_score(audio),
                'spectral_balance': self._calculate_balance_score(spectral_bands)
            }

            # Calculate spectral balance score
            spectral_metrics = {
                'band_balance_score': self._calculate_spectral_balance_score(spectral_bands)
            }

            # Compile analysis
            analysis = {
                'metrics': quality_metrics,
                'issues': self.common_issues.copy(),
                'language_specific': self.language_specific_issues.get(target_language, {}),
                'waveform_analysis': waveform_analysis,
                'spectral_analysis': {
                    'frequency_bands': spectral_bands,
                    'spectral_metrics': spectral_metrics
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
    def _calculate_spectral_balance_score(self, spectral_bands: Dict[str, float]) -> float:
        """Calculate spectral balance score based on energy distribution."""
        try:
            if not spectral_bands or not all(k in spectral_bands for k in ['low', 'mid', 'high']):
                return 0.0
            
            # Calculate relative energies
            total_energy = sum(spectral_bands.values())
            if total_energy == 0:
                return 0.0
            
            ratios = {k: v/total_energy for k, v in spectral_bands.items()}
        
            # Ideal ratios for speech
            ideal_ratios = {'low': 0.3, 'mid': 0.5, 'high': 0.2}
        
            # Calculate deviation from ideal
            deviation = sum(abs(ratios[k] - ideal_ratios[k]) for k in ratios)
        
            # Convert to 1-5 score (lower deviation is better)
            score = 5.0 * (1.0 - deviation)
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Spectral balance score calculation failed: {str(e)}")
            return 0.0

    def _measure_metallic_resonance(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Detect and measure metallic resonance artifacts common in neural speech synthesis.
        """
        try:
            # Convert to frequency domain with high resolution
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=512,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            
            # Analyze harmonic structure
            harmonic_peaks = self._find_harmonic_peaks(mag)
            peak_regularity = self._analyze_peak_regularity(harmonic_peaks)
            
            # Measure characteristics specific to metallic resonance
            measurements = {
                'harmonic_regularity': float(peak_regularity),
                'spectral_spikes': self._count_spectral_spikes(mag),
                'resonance_bands': self._identify_resonance_bands(mag),
                'temporal_stability': self._analyze_resonance_stability(mag)
            }
            
            # Generate detailed description
            description = []
            if measurements['harmonic_regularity'] > 0.8:
                description.append("Highly regular harmonic structure suggesting artificial resonance")
            if measurements['spectral_spikes'] > 10:
                description.append("Multiple sharp spectral peaks indicating metallic artifacts")
            if len(measurements['resonance_bands']) > 3:
                description.append("Multiple resonance bands contributing to synthetic timbre")
                
            return {
                'measurements': measurements,
                'severity': self._calculate_metallic_severity(measurements),
                'description': ". ".join(description),
                'temporal_evolution': self._track_resonance_evolution(mag),
                'frequency_distribution': self._analyze_resonance_distribution(mag)
            }
            
        except Exception as e:
            logger.error(f"Metallic resonance analysis failed: {str(e)}")
            return {}

    def diagnose_translation_quality(self, 
                                   source_audio: torch.Tensor,
                                   translated_audio: torch.Tensor,
                                   target_language: str) -> Dict[str, Any]:
        """
        Comprehensive diagnosis of translation quality issues with detailed comparisons.
        """
        try:
            diagnosis = {
                'source_analysis': self._analyze_source_characteristics(source_audio),
                'translation_analysis': self._analyze_translation_characteristics(translated_audio),
                'comparative_analysis': self._compare_audio_characteristics(
                    source_audio, 
                    translated_audio
                ),
                'translation_artifacts': self._analyze_translation_artifacts(translated_audio),
                'language_specific_issues': self._analyze_language_specific_issues(
                    translated_audio,
                    target_language
                ),
                'prosody_transfer': self._analyze_prosody_transfer(
                    source_audio,
                    translated_audio
                )
            }
            
            # Generate problem areas and recommendations
            diagnosis['problem_areas'] = self._identify_problem_areas(diagnosis)
            diagnosis['recommendations'] = self._generate_quality_recommendations(diagnosis)
            
            # Add detailed natural language description
            diagnosis['detailed_assessment'] = self._generate_translation_assessment(
                diagnosis,
                target_language
            )
            
            # Generate visualization data
            diagnosis['visualization_data'] = self._prepare_diagnostic_visualizations(
                source_audio,
                translated_audio,
                diagnosis
            )
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Translation quality diagnosis failed: {str(e)}")
            return {}

    def _generate_translation_assessment(self, 
                                      diagnosis: Dict[str, Any],
                                      target_language: str) -> str:
        """
        Generate a detailed, natural language assessment of translation quality.
        """
        assessment_points = []
        
        # Overall quality assessment
        quality_level = self._assess_overall_quality(diagnosis)
        assessment_points.append(f"Overall Translation Quality: {quality_level.name}")
        
        # Detailed artifact analysis
        if diagnosis['translation_artifacts']['voice_synthesis_issues']['robotic_artifacts']['detected']:
            severity = diagnosis['translation_artifacts']['voice_synthesis_issues']['robotic_artifacts']['severity']
            assessment_points.append(
                f"Detected robotic artifacts with severity {severity:.2f}/1.0. "
                "These manifest as metallic resonances and unnatural harmonic structures in the voice."
            )
            
        # Prosody assessment
        prosody_issues = diagnosis['translation_artifacts']['voice_synthesis_issues']['prosody_artifacts']
        if prosody_issues['detected']:
            assessment_points.append(
                "Prosody issues detected: " + 
                ". ".join(prosody_issues['issues'])
            )
            
        # Language-specific assessment
        lang_issues = diagnosis['language_specific_issues']
        if lang_issues:
            assessment_points.append(
                f"Language-specific issues for {target_language}: " +
                self._describe_language_issues(lang_issues, target_language)
            )
            
        # Recommendations
        if diagnosis['recommendations']:
            assessment_points.append(
                "Recommendations for improvement:\n- " +
                "\n- ".join(diagnosis['recommendations'])
            )
            
        return "\n\n".join(assessment_points)

    def _prepare_diagnostic_visualizations(self,
                                        source_audio: torch.Tensor,
                                        translated_audio: torch.Tensor,
                                        diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare comprehensive visualization data for diagnostic purposes.
        """
        try:
            visualizations = {
                'waveform_comparison': {
                    'source': source_audio.squeeze().numpy(),
                    'translated': translated_audio.squeeze().numpy(),
                    'alignment_points': self._find_alignment_points(
                        source_audio,
                        translated_audio
                    ),
                    'problem_areas': self._mark_problem_areas(
                        translated_audio,
                        diagnosis
                    )
                },
                'spectral_analysis': {
                    'source': self._generate_spectral_visualization(source_audio),
                    'translated': self._generate_spectral_visualization(translated_audio),
                    'differences': self._compute_spectral_differences(
                        source_audio,
                        translated_audio
                    )
                },
                'prosody_visualization': {
                    'pitch_contours': self._extract_pitch_contours(
                        source_audio,
                        translated_audio
                    ),
                    'energy_contours': self._extract_energy_contours(
                        source_audio,
                        translated_audio
                    ),
                    'rhythm_patterns': self._visualize_rhythm_patterns(
                        source_audio,
                        translated_audio
                    )
                },
                'quality_metrics': self._visualize_quality_metrics(diagnosis)
            }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Diagnostic visualization preparation failed: {str(e)}")
            return {}
    
    def analyze_neural_synthesis_artifacts(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Specialized analysis of neural speech synthesis artifacts and characteristics.
        """
        try:
            neural_analysis = {
                'voice_coherence': {
                    'formant_stability': self._analyze_formant_stability(audio),
                    'speaker_consistency': self._analyze_speaker_consistency(audio),
                    'voice_breaks': self._detect_voice_breaks(audio),
                    'timbre_continuity': self._analyze_timbre_continuity(audio)
                },
                'synthesis_artifacts': {
                    'attention_artifacts': self._detect_attention_artifacts(audio),
                    'oversmoothing': self._analyze_oversmoothing(audio),
                    'phoneme_boundary_issues': self._analyze_phoneme_boundaries(audio),
                    'artificial_resonances': self._detect_artificial_resonances(audio)
                },
                'naturalness_metrics': {
                    'micro_prosody': self._analyze_micro_prosody(audio),
                    'phonetic_timing': self._analyze_phonetic_timing(audio),
                    'coarticulation': self._analyze_coarticulation(audio),
                    'voice_quality_variations': self._analyze_voice_quality_variations(audio)
                },
                'error_patterns': {
                    'systematic_errors': self._identify_systematic_errors(audio),
                    'contextual_errors': self._analyze_contextual_errors(audio),
                    'model_specific_artifacts': self._identify_model_artifacts(audio)
                }
            }

            # Generate detailed descriptions for each category
            neural_analysis['detailed_descriptions'] = {
                'coherence_description': self._describe_voice_coherence(
                    neural_analysis['voice_coherence']
                ),
                'artifact_description': self._describe_synthesis_artifacts(
                    neural_analysis['synthesis_artifacts']
                ),
                'naturalness_description': self._describe_naturalness_metrics(
                    neural_analysis['naturalness_metrics']
                ),
                'error_description': self._describe_error_patterns(
                    neural_analysis['error_patterns']
                )
            }

            return neural_analysis
            
        except Exception as e:
            logger.error(f"Neural synthesis analysis failed: {str(e)}")
            return {}

    def generate_troubleshooting_report(self, 
                                      source_audio: torch.Tensor,
                                      translated_audio: torch.Tensor,
                                      target_language: str) -> Dict[str, Any]:
        """
        Generate comprehensive troubleshooting report with actionable insights.
        """
        try:
            # Perform all analyses
            neural_issues = self.analyze_neural_synthesis_artifacts(translated_audio)
            translation_issues = self._analyze_translation_artifacts(translated_audio)
            language_issues = self.analyze_language_specific_features(translated_audio, target_language)
            
            # Identify primary issues
            primary_issues = self._identify_primary_issues(
                neural_issues,
                translation_issues,
                language_issues
            )
            
            # Generate troubleshooting recommendations
            recommendations = self._generate_detailed_recommendations(
                primary_issues,
                target_language
            )
            
            # Create comprehensive report
            report = {
                'summary': {
                    'quality_assessment': self._assess_overall_quality(translated_audio),
                    'primary_issues': primary_issues,
                    'critical_problems': self._identify_critical_problems(primary_issues),
                    'quick_fixes': self._suggest_quick_fixes(primary_issues)
                },
                'detailed_analysis': {
                    'neural_synthesis': neural_issues,
                    'translation': translation_issues,
                    'language_specific': language_issues
                },
                'troubleshooting_guide': {
                    'recommended_actions': recommendations,
                    'priority_order': self._prioritize_fixes(recommendations),
                    'expected_improvements': self._predict_improvements(recommendations)
                },
                'technical_details': {
                    'error_patterns': self._analyze_error_patterns(translated_audio),
                    'quality_metrics': self._calculate_quality_metrics(translated_audio),
                    'performance_indicators': self._analyze_performance_indicators(translated_audio)
                }
            }
            
            # Add natural language descriptions
            report['narrative'] = self._generate_narrative_description(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Troubleshooting report generation failed: {str(e)}")
            return {}

    def _generate_narrative_description(self, report: Dict[str, Any]) -> str:
        """
        Generate a detailed narrative description of the audio issues and recommendations.
        """
        narrative = []
        
        # Overall quality assessment
        narrative.append(f"Overall Quality Assessment: {report['summary']['quality_assessment']}")
        
        # Describe primary issues
        if report['summary']['primary_issues']:
            narrative.append("\nPrimary Issues Detected:")
            for issue in report['summary']['primary_issues']:
                narrative.append(f"- {self._describe_issue_details(issue)}")
        
        # Critical problems
        if report['summary']['critical_problems']:
            narrative.append("\nCritical Problems Requiring Immediate Attention:")
            for problem in report['summary']['critical_problems']:
                narrative.append(f"- {self._describe_critical_problem(problem)}")
        
        # Detailed recommendations
        if report['troubleshooting_guide']['recommended_actions']:
            narrative.append("\nRecommended Actions in Priority Order:")
            for action in report['troubleshooting_guide']['recommended_actions']:
                narrative.append(f"- {self._describe_action_details(action)}")
        
        # Expected improvements
        if report['troubleshooting_guide']['expected_improvements']:
            narrative.append("\nExpected Improvements After Fixes:")
            for improvement in report['troubleshooting_guide']['expected_improvements']:
                narrative.append(f"- {self._describe_improvement(improvement)}")
        
        return "\n".join(narrative)

    def export_diagnostic_data(self, report: Dict[str, Any], export_path: str) -> None:
        """
        Export all diagnostic data and visualizations to a structured format.
        """
        try:
            export_data = {
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'visualizations': self._generate_all_visualizations(report),
                'metrics': self._compile_all_metrics(report),
                'recommendations': self._compile_all_recommendations(report)
            }
            
            # Save to file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Diagnostic data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
    
    def _calculate_robotic_score(self, audio: torch.Tensor) -> float:
        """Calculate roboticness score (1-5 scale) based on harmonic regularity"""
        try:
            # Convert to frequency domain
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
        
            # Calculate harmonic regularity
            harmonic_peaks = torch.max(mag, dim=1)[0]
            peak_variance = torch.var(harmonic_peaks)
        
            # More robotic speech has very regular harmonics (lower variance)
            score = 5.0 - (4.0 * (1.0 - peak_variance / torch.mean(harmonic_peaks)))
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Robotic score calculation failed: {str(e)}")
            return 0.0

    def _calculate_pronunciation_score(self, audio: torch.Tensor) -> float:
        """Calculate pronunciation score based on spectral clarity and formant strength"""
        try:
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
        
            # Analyze formant regions (approximately 500-3500 Hz)
            formant_region = mag[:, 50:350]  # Adjust indices based on your sampling rate
            formant_strength = torch.mean(formant_region)
        
            # Normalize to 1-5 scale
            score = 1.0 + 4.0 * (formant_strength / torch.max(mag))
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Pronunciation score calculation failed: {str(e)}")
            return 0.0

    def _calculate_clarity_score(self, audio: torch.Tensor) -> float:
        """Calculate clarity score based on signal-to-noise ratio and spectral contrast"""
        try:
            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(audio ** 2))
        
            # Calculate spectral contrast
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
            return_complex=True
            )
            mag = torch.abs(spec)
            contrast = torch.mean(torch.max(mag, dim=1)[0] / torch.mean(mag, dim=1))
        
            # Combine metrics
            score = 1.0 + 2.0 * float(rms) + 2.0 * float(contrast / 10.0)
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Clarity score calculation failed: {str(e)}")
            return 0.0

    def _calculate_noise_score(self, audio: torch.Tensor) -> float:
        """Calculate noise score based on signal-to-noise ratio"""
        try:
            # Calculate signal power
            signal_power = torch.mean(audio ** 2)
        
            # Estimate noise floor from silent regions
            sorted_amplitudes = torch.sort(torch.abs(audio.squeeze()))[0]
            noise_floor = torch.mean(sorted_amplitudes[:int(len(sorted_amplitudes)*0.1)]) ** 2
        
            # Calculate SNR and convert to score
            if noise_floor > 0:
                snr = 10 * torch.log10(signal_power / noise_floor)
                score = 1.0 + 4.0 * (torch.clamp(snr, 0, 50) / 50)
            else:
                score = 5.0
            
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Noise score calculation failed: {str(e)}")
            return 0.0

    def _calculate_consistency_score(self, audio: torch.Tensor) -> float:
        """Calculate voice consistency score based on amplitude and spectral stability"""
        try:
            # Analyze amplitude stability
            segments = torch.split(audio.squeeze(), 2048)
            segment_rms = torch.tensor([torch.sqrt(torch.mean(seg ** 2)) for seg in segments if len(seg) == 2048])
            amplitude_stability = 1.0 - torch.std(segment_rms) / torch.mean(segment_rms)
        
            # Analyze spectral stability
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            mag = torch.abs(spec)
            spectral_stability = 1.0 - torch.std(torch.mean(mag, dim=1)) / torch.mean(mag)
        
            # Combine metrics
            score = 1.0 + 4.0 * (amplitude_stability + spectral_stability) / 2
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Consistency score calculation failed: {str(e)}")
            return 0.0

    def _calculate_balance_score(self, spectral_bands: Dict[str, float]) -> float:
        """Calculate spectral balance score based on energy distribution"""
        try:
            if not spectral_bands or not all(k in spectral_bands for k in ['low', 'mid', 'high']):
                return 0.0
            
            # Calculate relative energies
            total_energy = sum(spectral_bands.values())
            if total_energy == 0:
                return 0.0
            
            ratios = {k: v/total_energy for k, v in spectral_bands.items()}
        
            # Ideal ratios for speech
            ideal_ratios = {'low': 0.3, 'mid': 0.5, 'high': 0.2}
        
            # Calculate deviation from ideal
            deviation = sum(abs(ratios[k] - ideal_ratios[k]) for k in ratios)
        
            # Convert to 1-5 score (lower deviation is better)
            score = 5.0 * (1.0 - deviation)
            return float(max(1.0, min(5.0, score)))
        
        except Exception as e:
            logger.error(f"Balance score calculation failed: {str(e)}")
            return 0.0