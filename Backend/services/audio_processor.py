import os
import torch
import torchaudio
import logging
import numpy as np
import scipy.signal
import scipy.stats
import timex
from pathlib import Path
from typing import Dict, Any, Tuple, Union

from .diagnostics import AudioDiagnostics

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    State-of-the-art audio processor with language-specific optimizations
    and comprehensive speech enhancement capabilities.
    """
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}
    MAX_AUDIO_LENGTH = 300  # seconds
    SAMPLE_RATE = 16000

    # Enhanced language-specific processing parameters optimized for speech recognition
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
            'speech_boost': 1.15,         # Added for speech enhancement
            'band_multipliers': {
                'sub_bass': (0.0, 0.03, 0.95),     # 20-60 Hz
                'bass': (0.03, 0.125, 1.0),        # 60-250 Hz
                'low_mid': (0.125, 0.25, 1.1),     # 250-500 Hz
                'mid': (0.25, 1.0, 1.15),          # 500-2000 Hz - Speech fundamental
                'high_mid': (1.0, 2.0, 1.05),      # 2000-4000 Hz - Speech clarity
                'presence': (2.0, 3.0, 0.95),      # 4000-6000 Hz
                'brilliance': (3.0, 10.0, 0.9)     # 6000+ Hz
            },
            'spectral_tilt': -1.5,
            'phase_coherence': 0.7
        },
        'deu': {  # German (similar structure for other languages...)
            'noise_reduction': 1.2,
            'compression_threshold': 0.4,
            'compression_ratio': 1.6,
            'brightness': 0.95,
            'clarity': 1.1,
            'formant_boost': 1.1,
            'high_freq_damping': 0.85,
            'temporal_smoothing': 0.9,
            'speech_boost': 1.2,
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
        }
    }
    # Add remaining language parameters here...

    def __init__(self) -> None:
        """Initialize audio processor with diagnostics capability"""
        try:
            self.diagnostics = AudioDiagnostics()
        except Exception as e:
            logger.error(f"Failed to initialize AudioDiagnostics: {str(e)}")
            self.diagnostics = None
    
    def is_valid_audio(self, audio: torch.Tensor) -> bool:
        """Check if audio data is valid with enhanced validation"""
        try:
            if not isinstance(audio, torch.Tensor):
                return False
                
            valid = (
                not torch.isnan(audio).any() and
                not torch.isinf(audio).any() and
                audio.abs().max() > 0 and
                audio.shape[1] > self.SAMPLE_RATE * 0.1  # At least 100ms of audio
            )
            
            # Additional speech-specific validation
            if valid:
                # Check for reasonable audio levels
                rms = torch.sqrt(torch.mean(audio ** 2))
                if rms < 1e-6 or rms > 1.0:
                    logger.warning("Audio levels outside optimal range for speech")
                    valid = False
                
                # Check for DC offset
                dc_offset = torch.mean(audio)
                if abs(dc_offset) > 0.1:
                    logger.warning("Significant DC offset detected")
                    valid = False
            
            return valid
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
        """Speech-optimized audio preprocessing"""
        try:
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
            audio = audio - torch.mean(audio, dim=1, keepdim=True)
            
            # Pre-emphasis filter for speech clarity
            pre_emphasis = 0.97
            audio = torch.cat([audio[:, :1], audio[:, 1:] - pre_emphasis * audio[:, :-1]], dim=1)
            
            # Handle silence gaps with speech-optimized thresholds
            rms = torch.sqrt(torch.mean(audio ** 2, dim=1, keepdim=True))
            silence_threshold = rms * 0.1
            is_silence = torch.abs(audio) < silence_threshold
            
            silence_segments = torch.nonzero(is_silence.squeeze(0))
            if silence_segments.dim() > 0:
                segments = torch.split(silence_segments, 1600)
                for segment in segments:
                    if segment.numel() > 800:
                        segment = segment[:800]
                        audio[0, segment] = audio[0, segment[:800]] * 0.01  # Preserve some silence
            
            # Gentle noise gate
            noise_gate_mask = (torch.abs(audio) > silence_threshold).float()
            audio = audio * noise_gate_mask
            
            # Normalize with speech-optimized headroom
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                target_peak = 0.95  # Increased for better speech recognition
                audio = audio * (target_peak / peak)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            return audio
        
    def detect_background_music(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Check for presence of background music with enhanced detection.
        """
        try:
            # Convert to spectrogram with overlapping windows for better frequency resolution
            f, t, spec = scipy.signal.spectrogram(
                audio_data, 
                fs=16000, 
                nperseg=2048,
                noverlap=1024,  # 50% overlap for better temporal resolution
                window='hann'
            )
    
            # 1. Harmonic Content Analysis
            spec_mean = np.mean(spec, axis=1)
            spectral_flatness = scipy.stats.gmean(spec_mean + 1e-10) / (np.mean(spec_mean) + 1e-10)
    
            # 2. Frequency Band Analysis with music-specific ranges
            bass_band = np.mean(spec[1:20, :])      # 0-156 Hz (bass)
            mid_band = np.mean(spec[20:50, :])      # 156-391 Hz (mid frequencies)
            presence_band = np.mean(spec[50:100, :]) # 391-781 Hz (presence)
    
            # Calculate band ratios (important for music detection)
            bass_mid_ratio = bass_band / (mid_band + 1e-10)
    
            # 3. Rhythmic Pattern Analysis
            energy_envelope = np.mean(spec, axis=0)
            peaks = scipy.signal.find_peaks(energy_envelope, distance=8)[0]  # Min distance for music beats
            rhythm_regularity = len(peaks) / len(t)  # Normalized peak count
    
            # 4. Temporal Stability
            temporal_variation = np.std(energy_envelope) / (np.mean(energy_envelope) + 1e-10)
            temporal_stability = 1.0 / (1.0 + temporal_variation)  # Now always positive, between 0 and 1

            # Combine all features with weighted scoring
            music_features = {
                'spectral_flatness': spectral_flatness * 0.2,
                'rhythm_regularity': rhythm_regularity * 0.3,
                'bass_presence': bass_mid_ratio * 0.3,
                'temporal_stability': temporal_stability * 0.2
            }
    
            music_score = sum(music_features.values())
    
            # Lower threshold and add confidence levels
            has_music = music_score > 0.25  # More sensitive threshold

            # 5. Speech vs Music Analysis
            speech_band = np.mean(spec[20:50, :])      # 156-391 Hz (key speech range)
            music_band = np.mean(spec[1:20, :])        # 0-156 Hz (typical music bass)
            speech_prominence = speech_band / (music_band + 1e-10)
        
            result = {
                'has_background_music': bool(has_music),
                'music_confidence': float(music_score),
                'feature_scores': {k: float(v) for k, v in music_features.items()},
                'speech_vs_music': {
                    'speech_prominence': float(speech_prominence),
                    'speech_band_energy': float(speech_band),
                    'music_band_energy': float(music_band)
                }
            }
    
            logger.debug(f"Music detection features: {result['feature_scores']}")
            logger.debug(f"Speech vs Music analysis: {result['speech_vs_music']}")
            return result

        except Exception as e:
            logger.error(f"Background music detection failed: {str(e)}")
            return {'has_background_music': False, 'music_confidence': 0.0}

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio with speech recognition focus"""
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
            
            # High-quality resampling optimized for speech
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
                # Weighted averaging for better speech preservation
                correlation = torch.sum(audio[0] * audio[1]) / torch.sqrt(
                    torch.sum(audio[0] ** 2) * torch.sum(audio[1] ** 2)
                )
                
                if correlation > 0.5:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                else:
                    mid = (audio[0] + audio[1]) / 2
                    side = (audio[0] - audio[1]) / 2
                    audio = (mid + 0.3 * side).unsqueeze(0)

            # Speech-optimized noise reduction
            spec = torch.stft(
                audio.squeeze(),
                n_fft=1024,  # Increased for better frequency resolution
                hop_length=256,
                win_length=1024,
                window=torch.hann_window(1024).to(audio.device),
                return_complex=True
            )
            
            mag = torch.abs(spec)
            phase = torch.angle(spec)
            
            noise_floor = torch.quantile(mag, 0.1, dim=1, keepdim=True)
            gain_mask = torch.clamp((mag - 1.5 * noise_floor) / (mag + 1e-8), 0, 1)
            
            # Enhance speech frequencies (1-4kHz)
            freq_bins = mag.shape[1]
            speech_range = (int(freq_bins * 0.0625), int(freq_bins * 0.25))  # ~1-4kHz
            gain_mask[:, speech_range[0]:speech_range[1]] *= 1.2
            
            enhanced_spec = mag * gain_mask * torch.exp(1j * phase)
            audio = torch.istft(
                enhanced_spec,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window=torch.hann_window(1024).to(audio.device)
            ).unsqueeze(0)

            # Final normalization
            max_amp = audio.abs().max()
            if max_amp > 0:
                audio = audio / max_amp * 0.95

            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")

    def apply_spectral_enhancement(self, audio: torch.Tensor, target_language: str) -> torch.Tensor:
        """Enhanced spectral processing optimized for speech"""
        try:
            if target_language not in self.LANGUAGE_PARAMS:
                logger.warning(f"No specific parameters for {target_language}, using defaults")
                target_language = 'fra'  # Use French as default

            params = self.LANGUAGE_PARAMS[target_language]
            
            # Apply multi-band processing with speech focus
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
            for spec in specs:
                mag = torch.abs(spec)
                phase = torch.angle(spec)
                
                # Enhanced frequency band processing
                freq_bins = mag.shape[1]
                for band_name, (start_ratio, end_ratio, multiplier) in params['band_multipliers'].items():
                    start_bin = max(0, int(freq_bins * start_ratio))
                    end_bin = min(freq_bins, int(freq_bins * end_ratio))
                    
                    if start_bin < end_bin:
                        # Apply speech-focused boost to mid and high-mid bands
                        if band_name in ['mid', 'high_mid']:
                            mag[:, start_bin:end_bin] *= multiplier * params.get('speech_boost', 1.0)
                        else:
                            mag[:, start_bin:end_bin] *= multiplier
                
                enhanced_specs.append(mag * torch.exp(1j * phase))

            # Multi-resolution synthesis
            audio_enhanced = torch.zeros_like(audio)
            weights = [0.2, 0.4, 0.4]  # Favor mid-resolution for speech
            
            for spec, weight, n_fft in zip(enhanced_specs, weights, [512, 1024, 2048]):
                audio_part = torch.istft(
                    spec,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(audio.device)
                )
                audio_enhanced[0, :len(audio_part)] += weight * audio_part

            # Apply speech-optimized compression
            rms = torch.sqrt(torch.mean(audio_enhanced ** 2))
            threshold = params['compression_threshold']
            ratio = params['compression_ratio']
            
            above_thresh = audio_enhanced.abs() > threshold
            gain_reduction = torch.zeros_like(audio_enhanced)
            gain_reduction[above_thresh] = -(audio_enhanced[above_thresh].abs() - threshold) * (1 - 1/ratio)
            
            audio_enhanced = audio_enhanced * torch.exp(gain_reduction)

            # Final normalization with speech-optimized headroom
            peak = torch.max(torch.abs(audio_enhanced))
            if peak > 0.95:
                audio_enhanced = audio_enhanced * (0.95 / peak)

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
        """Enhanced audio processing pipeline with speech optimization"""
        try:
            # Initial audio processing
            audio = self.process_audio(audio_path)
            logger.info("Base audio processing successful, proceeding with enhancement")
        
            # Apply preprocessing and get numpy for analysis
            audio = self.preprocess_audio(audio)
            audio_numpy = audio.squeeze().numpy()
        
            # Detect background music first
            music_analysis = self.detect_background_music(audio_numpy)
            logger.info(f"Background music analysis: {music_analysis}")
        
            # Apply spectral enhancement
            logger.info(f"Applying speech-optimized spectral enhancement for {target_language}")
            audio = self.apply_spectral_enhancement(audio, target_language)
        
            # Build full analysis
            full_analysis = {
                'background_music': music_analysis
            }
        
            # Run existing diagnostics if requested
            if return_diagnostics and self.diagnostics is not None:
                logger.info("Analyzing audio quality...")
                diagnostic_analysis = self.diagnostics.analyze_translation(audio, target_language)
            
                # Combine analyses
                full_analysis.update(diagnostic_analysis)
            
                return audio, full_analysis
        
            return audio if not return_diagnostics else (audio, full_analysis)
        
        except Exception as e:
            error_msg = f"Enhanced audio processing failed: {str(e)}"
            logger.error(error_msg)
            return torch.zeros((1, 1000)), {} if return_diagnostics else torch.zeros((1, 1000))