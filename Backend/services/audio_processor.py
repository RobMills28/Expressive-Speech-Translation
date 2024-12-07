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
            'formant_boost': 1.2,
            'high_freq_damping': 0.9,
            'temporal_smoothing': 0.8,
            'band_multipliers': {
                'sub_bass': (0.0, 0.03, 0.95),     # 20-60 Hz
                'bass': (0.03, 0.125, 1.0),        # 60-250 Hz
                'low_mid': (0.125, 0.25, 1.1),     # 250-500 Hz
                'mid': (0.25, 1.0, 1.15),          # 500-2000 Hz
                'high_mid': (1.0, 2.0, 1.05),      # 2000-4000 Hz
                'presence': (2.0, 3.0, 0.95),      # 4000-6000 Hz
                'brilliance': (3.0, 10.0, 0.9)     # 6000+ Hz
            },
            'spectral_tilt': -1.5,
            'phase_coherence': 0.7
        },
        'deu': {  # German
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.6,
            'brightness': 0.95,
            'clarity': 1.1,
            'formant_boost': 1.1,
            'high_freq_damping': 0.85,
            'temporal_smoothing': 0.9,
            'band_multipliers': {
                'sub_bass': (0.0, 0.03, 0.9),
                'bass': (0.03, 0.125, 1.0),
                'low_mid': (0.125, 0.25, 1.15),
                'mid': (0.25, 1.0, 1.1),
                'high_mid': (1.0, 2.0, 0.9),
                'presence': (2.0, 3.0, 0.85),
                'brilliance': (3.0, 10.0, 0.8)
            },
            'spectral_tilt': -2.0,
            'phase_coherence': 0.8
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
                'sub_bass': (0.0, 0.03, 0.9),
                'bass': (0.03, 0.125, 1.0),
                'low_mid': (0.125, 0.25, 1.1),
                'mid': (0.25, 1.0, 1.2),
                'high_mid': (1.0, 2.0, 1.1),
                'presence': (2.0, 3.0, 0.95),
                'brilliance': (3.0, 10.0, 0.9)
            },
            'spectral_tilt': -1.2,
            'phase_coherence': 0.75
        },
        'ita': {  # Italian
            'noise_reduction': 1.1,
            'compression_threshold': 0.35,
            'compression_ratio': 1.5,
            'brightness': 1.05,
            'clarity': 1.2,
            'formant_boost': 1.25,
            'high_freq_damping': 0.88,
            'temporal_smoothing': 0.82,
            'band_multipliers': {
                'sub_bass': (0.0, 0.03, 0.9),
                'bass': (0.03, 0.125, 1.0),
                'low_mid': (0.125, 0.25, 1.2),
                'mid': (0.25, 1.0, 1.15),
                'high_mid': (1.0, 2.0, 1.1),
                'presence': (2.0, 3.0, 1.0),
                'brilliance': (3.0, 10.0, 0.95)
            },
            'spectral_tilt': -1.0,
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
                'sub_bass': (0.0, 0.03, 0.9),
                'bass': (0.03, 0.125, 1.0),
                'low_mid': (0.125, 0.25, 1.15),
                'mid': (0.25, 1.0, 1.2),
                'high_mid': (1.0, 2.0, 1.1),
                'presence': (2.0, 3.0, 0.9),
                'brilliance': (3.0, 10.0, 0.85)
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
        """Check if audio data is valid"""
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
        """Validates audio file length and basic integrity"""
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
        Clean up input audio before processing.
    
        Args:
            audio (torch.Tensor): Input audio tensor (mono or stereo)
        
        Returns:
            torch.Tensor: Preprocessed audio tensor
        """
        try:
            # Input validation
            if not isinstance(audio, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")
            
            if audio.dim() == 0 or audio.size(0) == 0:
                raise ValueError("Empty audio tensor")
            
            # Ensure audio is in the correct shape [1, samples]
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            elif len(audio.shape) > 2:
                audio = audio.squeeze()
                if len(audio.shape) > 1:
                    audio = audio.mean(dim=0)
                audio = audio.unsqueeze(0)

            # Remove DC offset
            try:
                audio = audio - torch.mean(audio, dim=1, keepdim=True)
            except RuntimeError as e:
                logger.error(f"DC offset removal failed: {str(e)}")
                raise
            
            # Handle silence gaps
            try:
                rms = torch.sqrt(torch.mean(audio ** 2, dim=1, keepdim=True))
                silence_threshold = rms * 0.1
                is_silence = torch.abs(audio) < silence_threshold
            
                # Reduce silence duration while maintaining natural pauses
                is_silence = (torch.abs(audio) < silence_threshold).squeeze(0)  # Convert to 1D
                silence_segments = torch.nonzero(is_silence).squeeze()  # Get 1D indices
                if silence_segments.dim() > 0:  # Check if any silence found
                    segments = torch.split(silence_segments, 1600)
                    for segment in segments:
                        if segment.numel() > 800:  # Check length using numel()
                            segment = segment[:800]  # Limit segment length
                            audio[0, segment] = audio[0, segment[:800]]  # Apply to 2D tensor properly
                        
            except RuntimeError as e:
                logger.error(f"Silence processing failed: {str(e)}")
                raise
            
            # Gentle noise gate with validation
            try:
                noise_gate_mask = (torch.abs(audio) > silence_threshold).float()
                audio = audio * noise_gate_mask
            except RuntimeError as e:
                logger.error(f"Noise gate application failed: {str(e)}")
                raise
            
            # Normalize levels with validation
            try:
                peak = torch.max(torch.abs(audio))
                if peak > 0:
                    target_peak = 0.9
                    audio = audio * (target_peak / peak)
                
                # Validate final output
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    raise ValueError("Invalid values in processed audio")
                
            except RuntimeError as e:
                logger.error(f"Normalization failed: {str(e)}")
                raise
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            # Return original audio if processing fails
            return audio

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio with enhanced quality preservation"""
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
                    lowpass_filter_width=128,
                    rolloff=0.95,
                    resampling_method='kaiser_window',
                    beta=14.769656459379492
                )
                audio = resampler(audio)

            # Enhanced stereo to mono conversion
            if audio.shape[0] > 1:
                correlation = torch.sum(audio[0] * audio[1]) / torch.sqrt(
                    torch.sum(audio[0] ** 2) * torch.sum(audio[1] ** 2)
                )
                
                if correlation > 0.5:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                else:
                    mid = (audio[0] + audio[1]) / 2
                    side = (audio[0] - audio[1]) / 2
                    audio = (mid + 0.3 * side).unsqueeze(0)

            # Pre-emphasis
            pre_emphasis = 0.97
            audio = torch.cat([audio[:, :1], audio[:, 1:] - pre_emphasis * audio[:, :-1]], dim=1)

            # Adaptive normalization
            max_amp = audio.abs().max()
            if max_amp > 0:
                audio = audio / max_amp * 0.9

            # Noise reduction with spectral gating
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
            
            noise_floor = torch.median(mag, dim=1, keepdim=True)[0] * 0.1
            gain_mask = 1.0 - torch.exp(-(mag - noise_floor).clamp(min=0) / (noise_floor + 1e-8))
            
            enhanced_spec = mag * gain_mask * torch.exp(1j * phase)
            audio = torch.istft(
                enhanced_spec,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048).to(audio.device)
            ).unsqueeze(0)

            # De-emphasis
            audio = torch.cat([
                audio[:, :1],
                audio[:, 1:] + pre_emphasis * audio[:, :-1]
            ], dim=1)

            # Final normalization
            max_amp = audio.abs().max()
            if max_amp > 0:
                audio = audio / max_amp * 0.9

            logger.info(f"Processed audio shape: {audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")

    def apply_spectral_enhancement(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """Enhanced spectral processing with language-specific optimizations"""
        try:
            if torch.max(torch.abs(audio)) > 1.0:
                audio = audio / torch.max(torch.abs(audio))

            params = self.LANGUAGE_PARAMS[target_language]
            
            # Anti-aliasing filter
            nyquist = self.SAMPLE_RATE // 2
            cutoff = 0.95 * nyquist
            filter_kernel = torch.sinc(2 * cutoff * torch.linspace(-4, 4, 512)) * torch.hann_window(512)
            filter_kernel = filter_kernel / filter_kernel.sum()
            audio = torch.nn.functional.conv1d(
                audio.unsqueeze(0), 
                filter_kernel.view(1, 1, -1),
                padding='same'
            ).squeeze(0)

            # Dynamic range compression
            rms = torch.sqrt(torch.mean(audio ** 2))
            threshold = params['compression_threshold']
            ratio = params['compression_ratio']
            knee_width = threshold * 0.5
            
            gain_reduction = torch.zeros_like(audio)
            above_thresh_mask = audio.abs() > (threshold + knee_width/2)
            in_knee_mask = (audio.abs() > (threshold - knee_width/2)) & (audio.abs() <= (threshold + knee_width/2))
            
            gain_reduction[above_thresh_mask] = -(audio[above_thresh_mask].abs() - threshold) * (1 - 1/ratio)
            knee_factor = ((audio[in_knee_mask].abs() - (threshold - knee_width/2)) / knee_width) ** 2
            gain_reduction[in_knee_mask] = -knee_factor * (audio[in_knee_mask].abs() - threshold) * (1 - 1/ratio)
            
            audio = audio * torch.exp(gain_reduction)

            # Multi-band processing
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

            enhanced_specs = []
            for idx, spec in enumerate(specs):
                mag = torch.abs(spec)
                phase = torch.angle(spec)
                
                # Noise reduction
                noise_floor = torch.mean(mag[:, :10], dim=1, keepdim=True)
                mag = torch.max(mag - noise_floor * 1.5, torch.zeros_like(mag))
                
                # Frequency band processing
                freq_bins = mag.shape[1]
                nyquist = self.SAMPLE_RATE / 2
                
                for band_name, (start_ratio, end_ratio, multiplier) in params['band_multipliers'].items():
                    start_bin = max(0, int(freq_bins * start_ratio))
                    end_bin = min(freq_bins, int(freq_bins * end_ratio))
                    
                    if start_bin < end_bin:
                        mag[:, start_bin:end_bin] *= multiplier
                
                # Spectral smoothing
                mag = torch.nn.functional.conv2d(
                    mag.unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, 3, 3).to(mag.device) / 9,
                    padding=1
                ).squeeze(0).squeeze(0)
                
                enhanced_specs.append(mag * torch.exp(1j * phase))

            # Multi-resolution synthesis
            audio_enhanced = torch.zeros_like(audio)
            weights = [0.2, 0.4, 0.4]
            
            for spec, weight, n_fft in zip(enhanced_specs, weights, [512, 1024, 2048]):
                audio_part = torch.istft(
                    spec,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(audio.device)
                )
                audio_enhanced[0, :len(audio_part)] += weight * audio_part

            # Final limiting and warmth
            peak = torch.max(torch.abs(audio_enhanced))
            if peak > 0.95:
                audio_enhanced = audio_enhanced * (0.95 / peak)

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
        """Enhanced audio processing pipeline with quality control"""
        try:
            audio = self.process_audio(audio_path)
            logger.info("Base audio processing successful, proceeding with enhancement")
            
            audio = self.preprocess_audio(audio)
            
            logger.info(f"Applying spectral enhancement for {target_language}")
            audio = self.apply_spectral_enhancement(audio, target_language)
            
            if return_diagnostics and self.diagnostics is not None:
                logger.info("Analyzing audio quality...")
                analysis = self.diagnostics.analyze_translation(audio, target_language)
                
                # Reprocess if quality issues detected
                if (
                    analysis['waveform_analysis']['silence_percentage'] > 20 or
                    analysis['waveform_analysis']['rms_level'] < 0.1 or
                    analysis['issues'].get('clipping', False) or
                    analysis['issues'].get('choppy', False)
                ):
                    logger.warning("Quality issues detected, reprocessing...")
                    audio = self.preprocess_audio(audio)
                    audio = self.apply_spectral_enhancement(audio, target_language)
                    analysis = self.diagnostics.analyze_translation(audio, target_language)
                
                return audio, analysis
            
            return audio if not return_diagnostics else (audio, {})
            
        except Exception as e:
            error_msg = f"Enhanced audio processing failed: {str(e)}"
            logger.error(error_msg)
            return torch.zeros((1, 1000)), {} if return_diagnostics else torch.zeros((1, 1000))