import os
import torch
import torchaudio
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union

from .audio_diagnostics import AudioDiagnostics

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    State-of-the-art audio processor with language-specific optimizations
    and comprehensive speech enhancement capabilities.
    """
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}
    MAX_AUDIO_LENGTH = 300  # seconds
    SAMPLE_RATE = 16000

    # Enhanced language-specific processing parameters
    LANGUAGE_PARAMS = {
        'fra': {  # French
            'noise_reduction': 1.1,
            'compression_threshold': 0.35,
            'compression_ratio': 1.4,
            'brightness': 1.05,
            'clarity': 1.15,
            'formant_boost': 1.2,  # For French nasals
            'high_freq_damping': 0.9,  # Control sibilance
            'temporal_smoothing': 0.8,
            'band_multipliers': {
                'sub_bass': 0.95,     # 20-60 Hz
                'bass': 1.0,          # 60-250 Hz
                'low_mid': 1.1,       # 250-500 Hz
                'mid': 1.15,          # 500-2000 Hz - Critical for French vowels
                'high_mid': 1.05,     # 2000-4000 Hz
                'presence': 0.95,     # 4000-6000 Hz
                'brilliance': 0.9     # 6000+ Hz
            },
            'spectral_tilt': -1.5,    # Slight downward tilt for warmth
            'phase_coherence': 0.7    # Maintain vowel integrity
        },
        'deu': {  # German
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.6,
            'brightness': 0.95,
            'clarity': 1.1,
            'formant_boost': 1.1,     # Less than French
            'high_freq_damping': 0.85,
            'temporal_smoothing': 0.9,
            'band_multipliers': {
                'sub_bass': 0.9,
                'bass': 1.0,
                'low_mid': 1.15,      # Strong for German consonants
                'mid': 1.1,
                'high_mid': 0.9,
                'presence': 0.85,
                'brilliance': 0.8
            },
            'spectral_tilt': -2.0,    # More downward tilt for German
            'phase_coherence': 0.8    # Higher for consonant clarity
        },
        'spa': {  # Spanish
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.5,
            'brightness': 1.0,
            'clarity': 1.15,
            'formant_boost': 1.15,
            'high_freq_damping': 0.92,
            'temporal_smoothing': 0.85,
            'band_multipliers': {
                'sub_bass': 0.9,
                'bass': 1.0,
                'low_mid': 1.1,
                'mid': 1.2,           # Enhanced for Spanish vowels
                'high_mid': 1.1,
                'presence': 0.95,
                'brilliance': 0.9
            },
            'spectral_tilt': -1.2,    # Less tilt for Spanish brightness
            'phase_coherence': 0.75
        },
        'ita': {  # Italian
            'noise_reduction': 1.1,
            'compression_threshold': 0.35,
            'compression_ratio': 1.5,
            'brightness': 1.05,
            'clarity': 1.2,
            'formant_boost': 1.25,    # Strong for Italian vowels
            'high_freq_damping': 0.88,
            'temporal_smoothing': 0.82,
            'band_multipliers': {
                'sub_bass': 0.9,
                'bass': 1.0,
                'low_mid': 1.2,
                'mid': 1.15,
                'high_mid': 1.1,
                'presence': 1.0,
                'brilliance': 0.95
            },
            'spectral_tilt': -1.0,    # Minimal tilt for brightness
            'phase_coherence': 0.73
        },
        'por': {  # Portuguese
            'noise_reduction': 1.3,
            'compression_threshold': 0.45,
            'compression_ratio': 1.7,
            'brightness': 0.95,
            'clarity': 1.2,
            'formant_boost': 1.18,
            'high_freq_damping': 0.87,
            'temporal_smoothing': 0.88,
            'band_multipliers': {
                'sub_bass': 0.9,
                'bass': 1.0,
                'low_mid': 1.15,
                'mid': 1.2,
                'high_mid': 1.1,
                'presence': 0.9,
                'brilliance': 0.85
            },
            'spectral_tilt': -1.8,
            'phase_coherence': 0.76
        }
    }

    def __init__(self) -> None:
        """Initialize audio processor with diagnostics capability"""
        try:
            self.diagnostics = AudioDiagnostics()
        except Exception as e:
            logger.error(f"Failed to initialize AudioDiagnostics: {str(e)}")
            self.diagnostics = None
    
    def is_valid_audio(self, audio: torch.Tensor) -> bool:
        """
        Check if audio data is valid
        
        Args:
            audio (torch.Tensor): Audio tensor to validate
            
        Returns:
            bool: True if audio is valid, False otherwise
        """
        try:
            return (
                not torch.isnan(audio).any() and
                not torch.isinf(audio).any() and
                audio.abs().max() > 0 and
                audio.shape[1] > self.SAMPLE_RATE * 0.1  # At least 100ms of audio
            )
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            return False

    def validate_audio_length(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validates audio file length and basic integrity
        """
        try:
            if Path(audio_path).suffix.lower() not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {self.SUPPORTED_FORMATS}"

            if not os.path.exists(audio_path):
                return False, "Audio file not found"
                
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, "Audio file is empty"

            metadata = torchaudio.info(audio_path)
            
            if metadata.sample_rate <= 0:
                return False, "Invalid sample rate detected"
                
            if metadata.num_frames <= 0:
                return False, "No audio frames detected"

            duration = metadata.num_frames / metadata.sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration <= 0:
                return False, "Invalid audio duration"
                
            if duration > self.MAX_AUDIO_LENGTH:
                return False, f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({self.MAX_AUDIO_LENGTH}s)"

            return True, ""

        except Exception as e:
            error_msg = f"Error validating audio: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Clean up input audio before processing
        """
        try:
            # Remove DC offset
            audio = audio - torch.mean(audio, dim=1, keepdim=True)
            
            # Handle silence gaps
            rms = torch.sqrt(torch.mean(audio ** 2, dim=1, keepdim=True))
            silence_threshold = rms * 0.1
            is_silence = torch.abs(audio) < silence_threshold
            
            # Reduce silence duration while maintaining some natural pauses
            silence_segments = torch.nonzero(is_silence.squeeze())
            if len(silence_segments) > 0:
                segments = torch.split(silence_segments, 1600)  # ~100ms at 16kHz
                for segment in segments:
                    if len(segment) > 800:  # If silence is longer than 50ms
                        audio[:, segment[800:]] = audio[:, segment[:800]]  # Keep only first 50ms
            
            # Gentle noise gate
            audio = audio * (torch.abs(audio) > silence_threshold).float()
            
            # Normalize levels
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                target_peak = 0.9
                audio = audio * (target_peak / peak)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            return audio

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio with enhanced quality preservation."""
        try:
            logger.info(f"Loading audio from: {audio_path}")
        
            info = torchaudio.info(audio_path)
            logger.info(f"Audio info - Sample rate: {info.sample_rate}, Channels: {info.num_channels}")
        
            audio, orig_freq = torchaudio.load(audio_path)
        
        # Quality validation
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError("Audio contains invalid values")
            if audio.abs().max() == 0:
                raise ValueError("Audio is silent")
        
            logger.info(f"Original audio shape: {audio.shape}, Frequency: {orig_freq}Hz")
        
        # High-quality resampling
            if orig_freq != self.SAMPLE_RATE:
                logger.info(f"Resampling from {orig_freq}Hz to {self.SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_freq,
                    new_freq=self.SAMPLE_RATE,
                    lowpass_filter_width=128,  # Increased for better quality
                    rolloff=0.95,
                    resampling_method='kaiser_window',
                    beta=14.769656459379492
                )
                audio = resampler(audio)

        # Enhanced stereo to mono conversion if needed
            if audio.shape[0] > 1:
            # Use phase-aware mixing
                correlation = torch.sum(audio[0] * audio[1]) / torch.sqrt(
                    torch.sum(audio[0] ** 2) * torch.sum(audio[1] ** 2)
                )
            
                if correlation > 0.5:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                else:
                # Mid-side processing for better mono conversion
                    mid = (audio[0] + audio[1]) / 2
                    side = (audio[0] - audio[1]) / 2
                    audio = (mid + 0.3 * side).unsqueeze(0)

        # Pre-emphasis to improve high frequency response
            pre_emphasis = 0.97
            audio = torch.cat([audio[:, :1], audio[:, 1:] - pre_emphasis * audio[:, :-1]], dim=1)

        # Adaptive normalization
            max_amp = audio.abs().max()
            if max_amp > 0:
                audio = audio / max_amp * 0.9  # Leave headroom

        # Apply gentle noise reduction using spectral gating
            spec = torch.stft(
                audio.squeeze(),
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
        
            mag = torch.abs(spec)
            phase = torch.angle(spec)
        
        # Estimate noise floor
            noise_floor = torch.median(mag, dim=1, keepdim=True)[0] * 0.1
            gain_mask = 1.0 - torch.exp(-(mag - noise_floor).clamp(min=0) / (noise_floor + 1e-8))
        
        # Enhanced spectral reconstruction
            enhanced_spec = mag * gain_mask * torch.exp(1j * phase)
            audio = torch.istft(
                enhanced_spec,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048).to(audio.device)
            ).unsqueeze(0)

        # De-emphasis to balance pre-emphasis
            audio = torch.cat([
                audio[:, :1],
                audio[:, 1:] + pre_emphasis * audio[:, :-1]
            ], dim=1)

        # Final normalization with headroom
            max_amp = audio.abs().max()
            if max_amp > 0:
                audio = audio / max_amp * 0.9

            logger.info(f"Processed audio shape: {audio.shape}")
            return audio
        
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")

    def apply_spectral_enhancement(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """
        Modified spectral processing with improved audio quality controls
        """
        try:
            # Pre-processing normalization to prevent clipping
            if torch.max(torch.abs(audio)) > 1.0:
                audio = audio / torch.max(torch.abs(audio))

            params = self.LANGUAGE_PARAMS[target_language]
        
            # Anti-aliasing filter before processing
            nyquist = self.SAMPLE_RATE // 2
            cutoff = 0.95 * nyquist  # Slight rolloff to prevent aliasing
            filter_kernel = torch.sinc(2 * cutoff * torch.linspace(-4, 4, 512)) * torch.hann_window(512)
            filter_kernel = filter_kernel / filter_kernel.sum()
            audio = torch.nn.functional.conv1d(
                audio.unsqueeze(0), 
                filter_kernel.view(1, 1, -1),
                padding='same'
            ).squeeze(0)

            # Dynamic range compression with smoother knee
            rms = torch.sqrt(torch.mean(audio ** 2))
            threshold = params['compression_threshold']
            ratio = params['compression_ratio']
            knee_width = threshold * 0.5  # Wider knee for smoother transition
        
            gain_reduction = torch.zeros_like(audio)
            above_thresh_mask = audio.abs() > (threshold + knee_width/2)
            in_knee_mask = (audio.abs() > (threshold - knee_width/2)) & (audio.abs() <= (threshold + knee_width/2))
        
            # Smooth compression curve
            gain_reduction[above_thresh_mask] = -(audio[above_thresh_mask].abs() - threshold) * (1 - 1/ratio)
            knee_factor = ((audio[in_knee_mask].abs() - (threshold - knee_width/2)) / knee_width) ** 2
            gain_reduction[in_knee_mask] = -knee_factor * (audio[in_knee_mask].abs() - threshold) * (1 - 1/ratio)
        
            audio = audio * torch.exp(gain_reduction)

            # Multi-band processing with improved crossover
            specs = []
            for n_fft in [512, 1024, 2048]:
                spec = torch.stft(
                    audio[0],
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(audio.device),
                    return_complex=True
                )
                specs.append(spec)

            # Process each band separately with smoother transitions
            enhanced_specs = []
            for idx, spec in enumerate(specs):
                mag = torch.abs(spec)
                phase = torch.angle(spec)
            
                # Noise reduction with spectral subtraction
                noise_floor = torch.mean(mag[:, :10], dim=1, keepdim=True)  # Use first few frames as noise estimate
                mag = torch.max(mag - noise_floor * 1.5, torch.zeros_like(mag))
            
                # Apply band-specific processing
                freq_bins = mag.shape[1]
                for band_name, band_range in params['band_multipliers'].items():
                    start_bin = int(freq_bins * band_range[0])
                    end_bin = int(freq_bins * band_range[1])
                    mag[:, start_bin:end_bin] *= band_range[2]
            
                # Spectral smoothing
                mag = torch.nn.functional.conv2d(
                    mag.unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, 3, 3).to(mag.device) / 9,
                    padding=1
                ).squeeze(0).squeeze(0)
            
                enhanced_specs.append(mag * torch.exp(1j * phase))

            # Multi-resolution synthesis with weighted overlap
            audio_enhanced = torch.zeros_like(audio)
            weights = [0.2, 0.4, 0.4]  # More weight on higher resolutions
        
            for spec, weight, n_fft in zip(enhanced_specs, weights, [512, 1024, 2048]):
                audio_part = torch.istft(
                    spec,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(audio.device)
                )
                audio_enhanced[0, :len(audio_part)] += weight * audio_part

            # Final limiting and normalization
            peak = torch.max(torch.abs(audio_enhanced))
            if peak > 0.95:
                audio_enhanced = audio_enhanced * (0.95 / peak)

            # Add subtle warmth
            warmth = torch.tanh(audio_enhanced * 1.2) * 0.8
            audio_enhanced = audio_enhanced * 0.7 + warmth * 0.3

            return audio_enhanced

        except Exception as e:
            logger.error(f"Enhanced spectral processing failed: {str(e)}")
            return audio

    def process_audio_enhanced(
        self, 
        audio_path: str,
        target_language: str = 'fra',
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Enhanced audio processing pipeline with improved quality control
        """
        try:
            # Initial processing
            audio = self.process_audio(audio_path)
            
            # If we get here, process_audio succeeded
            logger.info("Base audio processing successful, proceeding with enhancement")
            
            # Clean up audio
            audio = self.preprocess_audio(audio)
            
            # First pass enhancement
            logger.info(f"Applying spectral enhancement for {target_language}")
            audio = self.apply_spectral_enhancement(audio, target_language)
            
            if return_diagnostics and self.diagnostics is not None:
                logger.info("Analyzing audio quality...")
                analysis = self.diagnostics.analyze_translation(audio, target_language)
                
                # Reprocess if quality issues detected
                if analysis['waveform_analysis']['silence_percentage'] > 20 or \
                   analysis['waveform_analysis']['rms_level'] < 0.1 or \
                   analysis['issues'].get('clipping', False) or \
                   analysis['issues'].get('choppy', False):
                    
                    logger.warning("Quality issues detected, reprocessing...")
                    # Additional cleanup pass
                    audio = self.preprocess_audio(audio)
                    # Second enhancement pass with more aggressive parameters
                    audio = self.apply_spectral_enhancement(audio, target_language)
                    # Get final analysis
                    analysis = self.diagnostics.analyze_translation(audio, target_language)
                
                return audio, analysis
            
            return audio if not return_diagnostics else (audio, {})
            
        except Exception as e:
            error_msg = f"Enhanced audio processing failed: {str(e)}"
            logger.error(error_msg)
            return torch.zeros((1, 1000)), {} if return_diagnostics else torch.zeros((1, 1000))