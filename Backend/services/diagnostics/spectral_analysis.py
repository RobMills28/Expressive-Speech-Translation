"""
Spectral analysis functionality for audio signal processing.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from .quality_metrics import FrequencyBand
import sys  # For terminal check
from typing import Dict, Any, List, Tuple, Optional, Iterable

logger = logging.getLogger(__name__)

class SpectralAnalyzer:
    """Handles spectral analysis of audio signals."""
    
    # Frequency bands with perceptual mappings for speech
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

    def __init__(self, sample_rate: int = 16000):
        """Initialize the analyzer with sample rate."""
        self.sample_rate = sample_rate
        logger.info(f"Initialized SpectralAnalyzer with sample rate: {sample_rate}Hz")

    def analyze_spectral_balance(self, audio: torch.Tensor) -> Dict[str, float]:
        """
        Analyze spectral balance by computing energy in different frequency bands.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, float]: Energy levels in different frequency bands
        """
        try:
            # Input validation
            if not isinstance(audio, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")

            if audio.numel() == 0:
                raise ValueError("Empty audio tensor")

            # Convert to mono if needed and make contiguous
            if audio.dim() > 1:
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.contiguous()

            # Compute STFT
            spec = torch.stft(
                audio.squeeze(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio.device),
                return_complex=True,
                center=True,
                pad_mode='reflect'
            )

            mag = torch.abs(spec)
            
            if mag.numel() == 0:
                raise ValueError("Empty magnitude spectrum")

            # Calculate energy in different bands with proper reshaping
            freq_bands = {}
            for band_name, band_info in self.FREQUENCY_BANDS.items():
                low_bin = max(1, int(band_info.low_freq * 4096 / self.sample_rate))
                high_bin = min(mag.shape[1], int(band_info.high_freq * 4096 / self.sample_rate))
                
                if low_bin < high_bin:
                    band_content = mag[:, low_bin:high_bin].reshape(-1, high_bin - low_bin)
                    freq_bands[band_name] = float(torch.mean(band_content).item())

            return freq_bands

        except Exception as e:
            logger.error(f"Spectral balance analysis failed: {str(e)}")
            return {band: 0.0 for band in self.FREQUENCY_BANDS.keys()}

    def analyze_voice_characteristics(self, mag: torch.Tensor) -> Dict[str, float]:
        """
        Analyze specific voice characteristics including raspiness and harmonic content.
        
        Args:
            mag (torch.Tensor): Magnitude spectrum tensor
            
        Returns:
            Dict[str, float]: Voice characteristics metrics
        """
        try:
            mag = mag.contiguous()
            
            # Focus on frequencies typical for voice characteristics
            raspy_range = self._get_frequency_range(mag, 2000, 4000)
            formant_range = self._get_frequency_range(mag, 300, 3400)
            harmonic_range = self._get_frequency_range(mag, 80, 1000)
            
            metrics = {
                'raspiness': self._calculate_raspiness(raspy_range),
                'harmonic_ratio': self._calculate_harmonic_ratio(harmonic_range),
                'formant_strength': self._calculate_formant_strength(formant_range),
                'spectral_tilt': self._calculate_spectral_tilt(mag),
                'voice_turbulence': self._calculate_voice_turbulence(raspy_range)
            }
            
            # Add confidence scores
            metrics.update(self._calculate_confidence_scores(metrics))
            
            return metrics

        except Exception as e:
            logger.error(f"Voice characteristics analysis failed: {str(e)}")
            return {
                'raspiness': 0.0,
                'harmonic_ratio': 0.0,
                'formant_strength': 0.0,
                'spectral_tilt': 0.0,
                'voice_turbulence': 0.0,
                'confidence_score': 0.0
            }

    def analyze_spectral_characteristics(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive spectral analysis providing detailed frequency-domain insights.
        
        Args:
            audio (torch.Tensor): Input audio tensor
            
        Returns:
            Dict[str, Any]: Comprehensive spectral analysis results
        """
        try:
            if audio.numel() == 0:
                raise ValueError("Empty audio tensor")

            # Make tensor contiguous and mono
            audio = audio.contiguous()
            if audio.dim() > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Multi-resolution analysis with voice-optimized windows
            window_sizes = [256, 512, 1024, 2048]
            spectral_data = {}

            for window_size in window_sizes:
                hop_length = window_size // 4  # 75% overlap
                
                # Compute STFT with proper padding
                spec = torch.stft(
                    audio.squeeze(),
                    n_fft=window_size,
                    hop_length=hop_length,
                    win_length=window_size,
                    window=torch.hann_window(window_size).to(audio.device),
                    return_complex=True,
                    center=True,
                    pad_mode='reflect'
                )
                
                mag = torch.abs(spec)
                
                if mag.numel() == 0:
                    continue

                # Comprehensive analysis at each resolution
                resolution_data = {
                    'band_analysis': self._analyze_frequency_bands(mag, window_size),
                    'voice_characteristics': self.analyze_voice_characteristics(mag),
                    'overall_characteristics': self._calculate_characteristics(mag),
                    'temporal_characteristics': self._analyze_temporal_characteristics(mag),
                    'harmonic_analysis': self._analyze_harmonics(mag, window_size),
                }

                spectral_data[f'resolution_{window_size}'] = resolution_data

            return spectral_data

        except Exception as e:
            logger.error(f"Spectral analysis failed: {str(e)}")
            return {}
    
    def _analyze_temporal_characteristics(self, mag: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze temporal characteristics with detailed metrics and visualizations.
        Args:
            mag (torch.Tensor): Magnitude spectrogram tensor
        Returns:
            Dict[str, Any]: Comprehensive temporal analysis results
        """
        try:
            # Get temporal envelope
            envelope = torch.mean(mag, dim=1)
            time_axis = torch.linspace(0, len(envelope) / self.sample_rate, len(envelope))
            
            # Temporal energy distribution
            energy_curve = torch.cumsum(envelope**2, dim=0) / torch.sum(envelope**2)
            
            # Onset detection
            onset_env = torch.diff(envelope)
            onset_env = torch.clamp(onset_env, min=0)
            peak_threshold = torch.mean(onset_env) + torch.std(onset_env)
            onset_peaks = (onset_env > peak_threshold).float()
            
            # Temporal statistics
            temporal_centroid = float(torch.sum(envelope * time_axis) / torch.sum(envelope))
            temporal_spread = float(torch.sqrt(
                torch.sum(envelope * (time_axis - temporal_centroid)**2) / torch.sum(envelope)
            ))
            
            # Calculate advanced metrics
            results = {
                'temporal_centroid': temporal_centroid,
                'temporal_spread': temporal_spread,
                'temporal_skewness': self._calculate_temporal_skewness(envelope, time_axis, temporal_centroid, temporal_spread),
                'temporal_kurtosis': self._calculate_temporal_kurtosis(envelope, time_axis, temporal_centroid, temporal_spread),
                'temporal_flatness': float(torch.exp(torch.mean(torch.log(envelope + 1e-8))) / (torch.mean(envelope) + 1e-8)),
                'onset_statistics': {
                    'onset_rate': float(torch.sum(onset_peaks) / len(envelope)),
                    'onset_strength_mean': float(torch.mean(onset_env).item()),
                    'onset_strength_std': float(torch.std(onset_env).item()),
                },
                'modulation': {
                    'mod_4hz': self._calculate_modulation_energy(envelope, 4),
                    'mod_8hz': self._calculate_modulation_energy(envelope, 8),
                    'mod_16hz': self._calculate_modulation_energy(envelope, 16),
                },
                'rhythm': {
                    'rhythm_strength': self._calculate_rhythm_strength(envelope),
                    'rhythm_regularity': self._calculate_rhythm_regularity(envelope),
                    'tempo_estimate': self._estimate_tempo(envelope)
                },
                'energy_distribution': {
                    'energy_quartiles': self._calculate_energy_quartiles(energy_curve),
                    'energy_concentration': self._calculate_energy_concentration(envelope),
                    'energy_decay': self._calculate_energy_decay(envelope)
                }
            }

            # Generate temporal visualizations if in interactive mode
            try:
                if sys.stdout.isatty():  # Check if running in terminal
                    self._plot_temporal_characteristics(
                        envelope.cpu().numpy(),
                        onset_peaks.cpu().numpy(),
                        time_axis.cpu().numpy(),
                        results
                    )
            except Exception as e:
                logger.warning(f"Visualization generation failed: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Temporal analysis failed: {str(e)}")
            return {
                'temporal_centroid': 0.0,
                'temporal_spread': 0.0,
                'temporal_skewness': 0.0,
                'temporal_kurtosis': 0.0,
                'temporal_flatness': 0.0,
                'onset_statistics': {'onset_rate': 0.0, 'onset_strength_mean': 0.0, 'onset_strength_std': 0.0},
                'modulation': {'mod_4hz': 0.0, 'mod_8hz': 0.0, 'mod_16hz': 0.0},
                'rhythm': {'rhythm_strength': 0.0, 'rhythm_regularity': 0.0, 'tempo_estimate': 0.0},
                'energy_distribution': {'energy_quartiles': [0.0, 0.0, 0.0], 'energy_concentration': 0.0, 'energy_decay': 0.0}
            }

    def _plot_temporal_characteristics(self, envelope: np.ndarray, onsets: np.ndarray, 
                                     time_axis: np.ndarray, metrics: Dict[str, Any]) -> None:
        """Generate ASCII visualization of temporal characteristics."""
        try:
            from ascii_graph import Pyasciigraph
            import ascii_graph.colors
            
            # Normalize envelope for display
            norm_envelope = envelope / np.max(envelope)
            
            # Create temporal envelope plot
            height = 10
            width = 80
            plot_data = np.interp(
                np.linspace(0, len(norm_envelope), width),
                np.arange(len(norm_envelope)),
                norm_envelope
            )
            
            print("\nTemporal Envelope Analysis:")
            print("=" * width)
            
            for i in range(height-1, -1, -1):
                line = ""
                for j in range(width):
                    if plot_data[j] > i/(height-1):
                        line += "█"
                    else:
                        line += " "
                print(line)
            
            print("=" * width)
            
            # Plot key metrics
            graph = Pyasciigraph()
            data = [
                ('Temporal Centroid', metrics['temporal_centroid']),
                ('Rhythm Strength', metrics['rhythm']['rhythm_strength']),
                ('Onset Rate', metrics['onset_statistics']['onset_rate']),
                ('Energy Concentration', metrics['energy_distribution']['energy_concentration'])
            ]
            
            for line in graph.graph('Key Temporal Metrics', data):
                print(line)
            
            # Print modulation analysis
            print("\nModulation Analysis:")
            print("-" * 40)
            mod_data = [
                ('4Hz Modulation', metrics['modulation']['mod_4hz']),
                ('8Hz Modulation', metrics['modulation']['mod_8hz']),
                ('16Hz Modulation', metrics['modulation']['mod_16hz'])
            ]
            for line in graph.graph('Modulation Energy', mod_data):
                print(line)
            
        except ImportError:
            logger.warning("ascii_graph not installed. Install with: pip install ascii_graph")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

    def _calculate_temporal_skewness(self, envelope: torch.Tensor, time_axis: torch.Tensor, 
                                   centroid: float, spread: float) -> float:
        """Calculate temporal skewness of the envelope."""
        try:
            normalized_time = (time_axis - centroid) / (spread + 1e-8)
            return float(torch.sum(envelope * normalized_time**3) / torch.sum(envelope))
        except Exception as e:
            logger.error(f"Temporal skewness calculation failed: {str(e)}")
            return 0.0

    def _calculate_temporal_kurtosis(self, envelope: torch.Tensor, time_axis: torch.Tensor, 
                                   centroid: float, spread: float) -> float:
        """Calculate temporal kurtosis of the envelope."""
        try:
            normalized_time = (time_axis - centroid) / (spread + 1e-8)
            return float(torch.sum(envelope * normalized_time**4) / torch.sum(envelope))
        except Exception as e:
            logger.error(f"Temporal kurtosis calculation failed: {str(e)}")
            return 0.0

    def _calculate_modulation_energy(self, envelope: torch.Tensor, freq: float) -> float:
        """Calculate modulation energy at specific frequency."""
        try:
            # Create modulation filter
            t = torch.arange(len(envelope)) / self.sample_rate
            mod_signal = torch.cos(2 * np.pi * freq * t)
            return float(torch.abs(torch.sum(envelope * mod_signal)))
        except Exception as e:
            logger.error(f"Modulation energy calculation failed: {str(e)}")
            return 0.0

    def _calculate_rhythm_strength(self, envelope: torch.Tensor) -> float:
        """Calculate overall rhythm strength."""
        try:
            # Convert to numpy for correlation
            env_np = envelope.cpu().numpy()
            autocorr = np.correlate(env_np, env_np, mode='full')
            center = len(autocorr) // 2
            autocorr = autocorr[center:]
        
            # Calculate rhythm strength ratio
            if len(autocorr) > 1 and autocorr[0] != 0:
                return float(np.max(autocorr[1:]) / autocorr[0])
            return 0.0
        
        except Exception as e:
            logger.error(f"Rhythm strength calculation failed: {str(e)}")
            return 0.0

    def _calculate_rhythm_regularity(self, envelope: torch.Tensor) -> float:
        """Calculate rhythm regularity from envelope."""
        try:
            # Convert to float and find peaks
            envelope = envelope.float()
            peaks = torch.zeros_like(envelope)
            peaks[1:-1] = ((envelope[1:-1] > envelope[:-2]) & 
                        (envelope[1:-1] > envelope[2:]))
            peak_indices = torch.where(peaks)[0]
        
            if len(peak_indices) < 2:
                return 0.0
            
            # Calculate intervals between peaks
            intervals = torch.diff(peak_indices)
            return float(1.0 - (torch.std(intervals.float()) / torch.mean(intervals.float())))
        except Exception as e:
            logger.error(f"Rhythm regularity calculation failed: {str(e)}")
            return 0.0

    def _estimate_tempo(self, envelope: torch.Tensor) -> float:
        """Estimate tempo from envelope in BPM."""
        try:
            # Convert to numpy for correlation
            env_np = envelope.cpu().numpy()
            autocorr = np.correlate(env_np, env_np, mode='full')
            center = len(autocorr) // 2
            autocorr = autocorr[center:]
        
            # Find peaks in autocorrelation
            peaks = np.zeros_like(autocorr)
            peaks[1:-1] = ((autocorr[1:-1] > autocorr[:-2]) & 
                        (autocorr[1:-1] > autocorr[2:]))
            peak_indices = np.where(peaks)[0]
        
            if len(peak_indices) < 2:
                return 0.0
            
            # Convert first major peak to BPM
            first_peak = peak_indices[1]  # Skip zero lag
            bpm = 60 * self.sample_rate / first_peak
            return float(bpm)
        except Exception as e:
            logger.error(f"Tempo estimation failed: {str(e)}")
            return 0.0

    def _analyze_frequency_bands(self, mag: torch.Tensor, window_size: int) -> Dict[str, Dict[str, float]]:
        """Analyze individual frequency bands with proper dimension handling."""
        try:
            band_data = {}
            mag = mag.contiguous()

            for band_name, band_info in self.FREQUENCY_BANDS.items():
                try:
                    # Calculate actual frequency range from Hz to bins with better validation
                    low_bin = max(1, int(band_info.low_freq * window_size / self.sample_rate))
                    high_bin = min(mag.shape[1] - 1, int(band_info.high_freq * window_size / self.sample_rate))
                
                    if low_bin >= high_bin:
                        logger.debug(f"Skipping {band_name} band: invalid bin range {low_bin}-{high_bin}")
                        continue

                    # Ensure valid dimension ranges
                    low_bin = min(low_bin, mag.shape[1] - 1)
                    high_bin = min(high_bin, mag.shape[1])
                
                    band_content = mag[:, low_bin:high_bin].reshape(-1, high_bin - low_bin)
                
                    if band_content.numel() == 0:
                        continue

                    # Keep your original detailed metrics
                    band_data[band_name] = {
                        'mean_energy': float(torch.mean(band_content).item()),
                        'peak_energy': float(torch.max(band_content).item()),
                        'energy_variation': float(torch.std(torch.mean(band_content, dim=1)).item()),
                        'spectral_slope': self._calculate_spectral_slope(band_content),
                        'temporal_stability': self._calculate_temporal_stability(band_content),
                        'frequency_range': f"{band_info.low_freq}-{band_info.high_freq}Hz"
                    }

                except Exception as e:
                    logger.debug(f"Error processing {band_name} band: {str(e)}")  # Changed to debug level
                    continue

            return band_data

        except Exception as e:
            logger.error(f"Frequency band analysis failed: {str(e)}")
            return {}

    def _calculate_characteristics(self, mag: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive spectral characteristics."""
        try:
            return {
                'spectral_centroid': self._calculate_spectral_centroid(mag),
                'spectral_spread': self._calculate_spectral_spread(mag),
                'spectral_skewness': self._calculate_spectral_skewness(mag),
                'spectral_flatness': self._calculate_spectral_flatness(mag),
                'spectral_rolloff': self._calculate_spectral_rolloff(mag),
                'spectral_entropy': self._calculate_spectral_entropy(mag)
            }
        except Exception as e:
            logger.error(f"Characteristics calculation failed: {str(e)}")
            return {
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'spectral_skewness': 0.0,
                'spectral_flatness': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_entropy': 0.0
            }

    def _get_frequency_range(self, mag: torch.Tensor, low_freq: int, high_freq: int) -> torch.Tensor:
        """Extract specific frequency range from magnitude spectrum."""
        try:
            low_bin = max(1, int(low_freq * mag.shape[1] / self.sample_rate))
            high_bin = min(mag.shape[1], int(high_freq * mag.shape[1] / self.sample_rate))
            return mag[:, low_bin:high_bin].reshape(-1, high_bin - low_bin)
        except Exception as e:
            logger.error(f"Failed to get frequency range {low_freq}-{high_freq}Hz: {str(e)}")
            return torch.zeros(1, 1)

    def _calculate_spectral_centroid(self, mag: torch.Tensor) -> float:
        """Calculate spectral centroid with robust error handling."""
        try:
            freqs = torch.linspace(0, 1, mag.shape[1], device=mag.device)
            norm = torch.sum(mag, dim=1, keepdim=True) + 1e-8
            centroid = torch.sum(mag * freqs.unsqueeze(0), dim=1) / norm.squeeze()
            return float(torch.mean(centroid).item())
        except Exception as e:
            logger.error(f"Spectral centroid calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_spread(self, mag: torch.Tensor) -> float:
        """Calculate spectral spread with robust error handling."""
        try:
            freqs = torch.linspace(0, 1, mag.shape[1], device=mag.device)
            centroid = self._calculate_spectral_centroid(mag)
            norm = torch.sum(mag, dim=1, keepdim=True) + 1e-8
            spread = torch.sqrt(
                torch.sum(mag * (freqs.unsqueeze(0) - centroid) ** 2, dim=1) / norm.squeeze()
            )
            return float(torch.mean(spread).item())
        except Exception as e:
            logger.error(f"Spectral spread calculation failed: {str(e)}")
            return 0.0

    def _calculate_temporal_characteristics(self, mag: torch.Tensor) -> Dict[str, float]:
        """Analyze temporal characteristics of the signal."""
        try:
            envelope = torch.mean(mag, dim=1)
            return {
                'temporal_centroid': float(torch.sum(envelope * torch.arange(len(envelope))) / torch.sum(envelope)),
                'temporal_flatness': float(torch.exp(torch.mean(torch.log(envelope + 1e-8))) / (torch.mean(envelope) + 1e-8)),
                'onset_strength': float(torch.mean(torch.diff(envelope).clamp(min=0)).item())
            }
        except Exception as e:
            logger.error(f"Temporal analysis failed: {str(e)}")
            return {'temporal_centroid': 0.0, 'temporal_flatness': 0.0, 'onset_strength': 0.0}

    def _calculate_confidence_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate confidence scores for voice analysis metrics."""
        try:
            # Normalize metrics to 0-1 range and weight them
            weights = {
                'raspiness': 0.3,
                'harmonic_ratio': 0.3,
                'formant_strength': 0.2,
                'spectral_tilt': 0.1,
                'voice_turbulence': 0.1
            }
            
            confidence = sum(
                weights.get(key, 0) * self._normalize_metric(value)
                for key, value in metrics.items()
                if key in weights
            )
            
            return {
                'confidence_score': float(confidence),
                'reliability_estimate': float(min(1.0, confidence * 1.2))
            }
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return {'confidence_score': 0.0, 'reliability_estimate': 0.0}

    def _normalize_metric(self, value: float) -> float:
        """Normalize metric to 0-1 range with sigmoid function."""
        try:
            return float(1 / (1 + np.exp(-value)))
        except Exception as e:
            logger.error(f"Metric normalization failed: {str(e)}")
            return 0.0

    def _calculate_spectral_skewness(self, mag: torch.Tensor) -> float:
        """Calculate spectral skewness with improved accuracy."""
        try:
            freqs = torch.linspace(0, 1, mag.shape[1], device=mag.device)
            centroid = self._calculate_spectral_centroid(mag)
            spread = self._calculate_spectral_spread(mag) + 1e-8
            norm = torch.sum(mag, dim=1, keepdim=True) + 1e-8
            skew = torch.sum(
                mag * ((freqs.unsqueeze(0) - centroid) / spread) ** 3, 
                dim=1
            ) / norm.squeeze()
            return float(torch.mean(skew).item())
        except Exception as e:
            logger.error(f"Spectral skewness calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_flatness(self, mag: torch.Tensor) -> float:
        """Calculate spectral flatness (Wiener entropy)."""
        try:
            eps = 1e-8
            mag_safe = mag + eps
            geometric_mean = torch.exp(torch.mean(torch.log(mag_safe), dim=1))
            arithmetic_mean = torch.mean(mag, dim=1) + eps
            flatness = geometric_mean / arithmetic_mean
            return float(torch.mean(flatness).item())
        except Exception as e:
            logger.error(f"Spectral flatness calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_rolloff(self, mag: torch.Tensor, percentile: float = 0.85) -> float:
        """Calculate frequency below which percentile of energy exists."""
        try:
            total_energy = torch.sum(mag, dim=1, keepdim=True)
            cumsum = torch.cumsum(mag, dim=1) / total_energy
            rolloff_bin = torch.sum(cumsum < percentile, dim=1)
            return float(torch.mean(rolloff_bin.float() / mag.shape[1]).item())
        except Exception as e:
            logger.error(f"Spectral rolloff calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_entropy(self, mag: torch.Tensor) -> float:
        """Calculate spectral entropy as a measure of spectral distribution."""
        try:
            eps = 1e-8
            prob = mag / (torch.sum(mag, dim=1, keepdim=True) + eps)
            entropy = -torch.sum(prob * torch.log2(prob + eps), dim=1)
            return float(torch.mean(entropy).item())
        except Exception as e:
            logger.error(f"Spectral entropy calculation failed: {str(e)}")
            return 0.0

    def _calculate_raspiness(self, mag: torch.Tensor) -> float:
        """Calculate voice raspiness metric."""
        try:
            # High frequency energy ratio indicates raspiness
            total_energy = torch.sum(mag)
            if total_energy == 0:
                return 0.0
            high_freq_ratio = torch.sum(mag[:, mag.shape[1]//2:]) / total_energy
            return float(high_freq_ratio.item())
        except Exception as e:
            logger.error(f"Raspiness calculation failed: {str(e)}")
            return 0.0

    def _calculate_harmonic_ratio(self, mag: torch.Tensor) -> float:
        """Calculate harmonic to noise ratio."""
        try:
            # Simplified HNR calculation
            peaks = torch.max(mag, dim=1)[0]
            mean = torch.mean(mag, dim=1)
            ratio = peaks / (mean + 1e-8)
            return float(torch.mean(ratio).item())
        except Exception as e:
            logger.error(f"Harmonic ratio calculation failed: {str(e)}")
            return 0.0

    def _calculate_formant_strength(self, mag: torch.Tensor) -> float:
        """Calculate formant strength as measure of vowel clarity."""
        try:
            # Find local maxima in the formant range
            diff = torch.diff(mag.mean(dim=0))
            formant_peaks = torch.sum((diff[:-1] > 0) & (diff[1:] < 0))
            return float(formant_peaks.item() / mag.shape[1])
        except Exception as e:
            logger.error(f"Formant strength calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_tilt(self, mag: torch.Tensor) -> float:
        """Calculate spectral tilt (overall slope of spectrum)."""
        try:
            freqs = torch.linspace(0, 1, mag.shape[1], device=mag.device)
            log_mag = torch.log(torch.mean(mag, dim=0) + 1e-8)
            coeffs = np.polyfit(freqs.cpu().numpy(), log_mag.cpu().numpy(), deg=1)
            return float(coeffs[0])
        except Exception as e:
            logger.error(f"Spectral tilt calculation failed: {str(e)}")
            return 0.0

    def _calculate_voice_turbulence(self, mag: torch.Tensor) -> float:
        """Calculate voice turbulence index."""
        try:
            # Measure of irregularity in high frequencies
            high_freq = mag[:, mag.shape[1]//2:]
            turbulence = torch.std(high_freq) / (torch.mean(high_freq) + 1e-8)
            return float(turbulence.item())
        except Exception as e:
            logger.error(f"Voice turbulence calculation failed: {str(e)}")
            return 0.0

    def _calculate_temporal_stability(self, band_content: torch.Tensor) -> float:
        """Calculate temporal stability of frequency band."""
        try:
            temporal_env = torch.mean(band_content, dim=1)
            stability = 1.0 - (torch.std(temporal_env) / (torch.mean(temporal_env) + 1e-8))
            return float(stability.item())
        except Exception as e:
            logger.error(f"Temporal stability calculation failed: {str(e)}")
            return 0.0

    def _calculate_spectral_slope(self, band_content: torch.Tensor) -> float:
        """Calculate spectral slope within frequency band."""
        try:
            mean_spectrum = torch.mean(band_content, dim=0)
            freqs = torch.linspace(0, 1, mean_spectrum.shape[0], device=band_content.device)
            coeffs = np.polyfit(freqs.cpu().numpy(), mean_spectrum.cpu().numpy(), deg=1)
            return float(coeffs[0])
        except Exception as e:
            logger.error(f"Spectral slope calculation failed: {str(e)}")
            return 0.0

    def _analyze_harmonics(self, mag: torch.Tensor, window_size: int) -> Dict[str, float]:
        """Analyze harmonic content of the signal."""
        try:
            # Convert to float before calculations
            mag = mag.float()
            mean_spectrum = torch.mean(mag, dim=0)
            peak_threshold = torch.mean(mean_spectrum) + torch.std(mean_spectrum)
            peaks = (mean_spectrum > peak_threshold).float()
        
        # Handle potential empty peaks
            peaks_indices = peaks.nonzero().squeeze()
            if peaks_indices.numel() > 1:  # Check if we have at least 2 peaks
                peak_diffs = torch.diff(peaks_indices)
                harmonic_spacing = float(torch.mean(peak_diffs).item())
                harmonic_regularity = float(torch.std(peak_diffs).item())
            else:
                harmonic_spacing = 0.0
                harmonic_regularity = 0.0
            
            return {
                'harmonic_density': float(torch.mean(peaks).item()),
                'harmonic_spacing': harmonic_spacing,
                'harmonic_regularity': harmonic_regularity
            }
        except Exception as e:
            logger.error(f"Harmonic analysis failed: {str(e)}")
            return {'harmonic_density': 0.0, 'harmonic_spacing': 0.0, 'harmonic_regularity': 0.0}
        
    def _calculate_energy_quartiles(self, energy_curve: torch.Tensor) -> List[float]:
        """Calculate energy quartiles from cumulative energy curve."""
        try:
            quartiles = []
            for q in [0.25, 0.5, 0.75]:
                idx = torch.searchsorted(energy_curve, q)
                quartiles.append(float(idx.item() / len(energy_curve)))
            return quartiles
        except Exception as e:
            logger.error(f"Energy quartiles calculation failed: {str(e)}")
            return [0.0, 0.0, 0.0]

    def _calculate_energy_concentration(self, envelope: torch.Tensor) -> float:
        """Calculate what fraction of time contains 80% of signal energy."""
        try:
            sorted_env = torch.sort(envelope, descending=True)[0]
            cumsum = torch.cumsum(sorted_env, dim=0)
            threshold = 0.8 * cumsum[-1]  # 80% of total energy
            frames = torch.searchsorted(cumsum, threshold)
            return float(1.0 - frames.item() / len(envelope))  # Higher value means more concentrated
        except Exception as e:
            logger.error(f"Energy concentration calculation failed: {str(e)}")
            return 0.0

    def _calculate_energy_decay(self, envelope: torch.Tensor) -> float:
        """Calculate how quickly energy decays over time."""
        try:
            # Convert to numpy for polyfit
            log_env = np.log10(envelope.cpu().numpy() + 1e-8)
            time_idx = np.arange(len(envelope)) / self.sample_rate
            slope = np.polyfit(time_idx, log_env, deg=1)[0]
            return float(slope)  # Negative values indicate faster decay
        except Exception as e:
            logger.error(f"Energy decay calculation failed: {str(e)}")
            return 0.0