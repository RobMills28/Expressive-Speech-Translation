# services/temporal_mapper.py
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch

logger = logging.getLogger(__name__)

class TemporalMapper:
    """
    Simple, robust temporal mapping that preserves natural speech characteristics
    without forcing exact timing alignment
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.min_significant_pause = 0.2  # 200ms minimum pause to consider
        self.max_stretch_factor = 1.5     # Maximum tempo stretching
        self.min_stretch_factor = 0.7     # Minimum tempo stretching
        
    def extract_timing_profile(self, audio: np.ndarray, word_timestamps: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Extract timing characteristics from audio
        Returns a simple profile of speech timing
        """
        duration = len(audio) / self.sample_rate
        
        if word_timestamps and len(word_timestamps) > 1:
            # Use word timestamps if available
            speaking_segments = []
            pause_segments = []
            
            for i, word in enumerate(word_timestamps):
                speaking_segments.append(word['end'] - word['start'])
                
                if i < len(word_timestamps) - 1:
                    pause_duration = word_timestamps[i + 1]['start'] - word['end']
                    if pause_duration > self.min_significant_pause:
                        pause_segments.append(pause_duration)
            
            total_speaking_time = sum(speaking_segments)
            total_pause_time = sum(pause_segments)
            
            return {
                'total_duration': duration,
                'speaking_time': total_speaking_time,
                'pause_time': total_pause_time,
                'speaking_ratio': total_speaking_time / duration if duration > 0 else 0.8,
                'average_pause': np.mean(pause_segments) if pause_segments else 0.3,
                'num_pauses': len(pause_segments),
                'speech_rate': len(word_timestamps) / total_speaking_time if total_speaking_time > 0 else 3.0
            }
        else:
            # Fallback to energy-based analysis
            return self._energy_based_profile(audio)
    
    def _energy_based_profile(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Fallback method using energy analysis when word timestamps aren't available
        """
        # Frame-based energy analysis
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Calculate frame energy
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        
        # Smooth energy curve
        energy_smooth = gaussian_filter1d(energy, sigma=2.0)
        
        # Voice activity detection
        threshold = np.percentile(energy_smooth, 30)
        voiced_frames = energy_smooth > threshold
        
        # Estimate speaking and pause segments
        speaking_ratio = np.mean(voiced_frames)
        
        # Find pause regions
        unvoiced_regions = ~voiced_frames
        pause_starts = np.where(np.diff(unvoiced_regions.astype(int)) == 1)[0]
        pause_ends = np.where(np.diff(unvoiced_regions.astype(int)) == -1)[0]
        
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            if len(pause_ends) < len(pause_starts):
                pause_ends = np.append(pause_ends, len(unvoiced_regions) - 1)
            if len(pause_starts) < len(pause_ends):
                pause_starts = np.insert(pause_starts, 0, 0)
            
            pause_durations = []
            for start, end in zip(pause_starts, pause_ends):
                duration = (end - start) * hop_length / self.sample_rate
                if duration > self.min_significant_pause:
                    pause_durations.append(duration)
        else:
            pause_durations = []
        
        duration = len(audio) / self.sample_rate
        
        return {
            'total_duration': duration,
            'speaking_time': duration * speaking_ratio,
            'pause_time': sum(pause_durations),
            'speaking_ratio': speaking_ratio,
            'average_pause': np.mean(pause_durations) if pause_durations else 0.3,
            'num_pauses': len(pause_durations),
            'speech_rate': speaking_ratio * 3.0  # Rough estimate
        }
    
    def apply_temporal_guidance(self, target_audio: np.ndarray, source_profile: Dict[str, Any]) -> np.ndarray:
        """
        Apply gentle temporal guidance to target audio based on source characteristics
        This is the main function that does the temporal mapping
        """
        target_duration = len(target_audio) / self.sample_rate
        source_duration = source_profile['total_duration']
        
        # Calculate target speaking ratio based on source but don't force it
        source_speaking_ratio = source_profile['speaking_ratio']
        target_speaking_ratio = self._estimate_speaking_ratio(target_audio)
        
        # If the ratios are very different, apply gentle adjustment
        ratio_difference = abs(source_speaking_ratio - target_speaking_ratio)
        
        if ratio_difference > 0.2:  # Only adjust if significantly different
            logger.info(f"Applying temporal guidance: source ratio {source_speaking_ratio:.2f}, target ratio {target_speaking_ratio:.2f}")
            
            # Apply gentle tempo stretching
            adjusted_audio = self._apply_gentle_stretching(target_audio, source_profile)
            
            # Apply natural pause enhancement
            final_audio = self._enhance_natural_pauses(adjusted_audio, source_profile)
            
            return final_audio
        else:
            logger.info("Target audio timing already natural, minimal adjustment needed")
            return self._enhance_natural_pauses(target_audio, source_profile)
    
    def _estimate_speaking_ratio(self, audio: np.ndarray) -> float:
        """
        Estimate speaking ratio of audio using energy analysis
        """
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        energy_smooth = gaussian_filter1d(energy, sigma=2.0)
        
        threshold = np.percentile(energy_smooth, 25)
        voiced_frames = energy_smooth > threshold
        
        return np.mean(voiced_frames)
    
    def _apply_gentle_stretching(self, audio: np.ndarray, source_profile: Dict[str, Any]) -> np.ndarray:
        """
        Apply gentle tempo stretching using phase vocoder
        """
        # Calculate desired stretch factor
        target_duration = len(audio) / self.sample_rate
        desired_ratio = source_profile['speaking_ratio']
        current_ratio = self._estimate_speaking_ratio(audio)
        
        # Calculate stretch factor but limit it to reasonable bounds
        if current_ratio > 0:
            stretch_factor = desired_ratio / current_ratio
            stretch_factor = np.clip(stretch_factor, self.min_stretch_factor, self.max_stretch_factor)
        else:
            stretch_factor = 1.0
        
        if abs(stretch_factor - 1.0) < 0.1:
            return audio  # No stretching needed
        
        logger.info(f"Applying gentle tempo stretch: factor = {stretch_factor:.2f}")
        
        # Use librosa's phase vocoder for high-quality time stretching
        stft = librosa.stft(audio, hop_length=512)
        stft_stretched = librosa.phase_vocoder(stft, rate=stretch_factor)
        stretched_audio = librosa.istft(stft_stretched, hop_length=512)
        
        return stretched_audio
    
    def _enhance_natural_pauses(self, audio: np.ndarray, source_profile: Dict[str, Any]) -> np.ndarray:
        """
        Enhance natural pauses in the audio based on source characteristics
        """
        # Find natural pause locations in the audio
        pause_locations = self._find_natural_pause_locations(audio)
        
        if not pause_locations:
            return audio
        
        # Determine how many pauses to enhance based on source
        source_pause_density = source_profile['num_pauses'] / source_profile['total_duration']
        target_duration = len(audio) / self.sample_rate
        desired_num_pauses = int(source_pause_density * target_duration)
        
        # Select the best pause locations to enhance
        if len(pause_locations) > desired_num_pauses:
            # Sort by confidence and take the top ones
            pause_locations.sort(key=lambda x: x['confidence'], reverse=True)
            pause_locations = pause_locations[:desired_num_pauses]
        
        # Apply pause enhancement
        enhanced_audio = self._apply_pause_enhancements(audio, pause_locations, source_profile)
        
        return enhanced_audio
    
    def _find_natural_pause_locations(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find natural locations where pauses could be enhanced
        """
        # Use energy dips and spectral characteristics
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        # Energy analysis
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        energy_smooth = gaussian_filter1d(energy, sigma=3.0)
        
        # Find energy minima
        minima_indices, properties = find_peaks(-energy_smooth, height=-np.percentile(energy_smooth, 40))
        
        pause_locations = []
        for idx in minima_indices:
            time_position = idx * hop_length / self.sample_rate
            
            # Calculate confidence based on how much of a dip this is
            local_window = slice(max(0, idx - 10), min(len(energy_smooth), idx + 10))
            local_mean = np.mean(energy_smooth[local_window])
            confidence = (local_mean - energy_smooth[idx]) / (local_mean + 1e-8)
            
            if confidence > 0.3:  # Minimum confidence threshold
                pause_locations.append({
                    'position': time_position,
                    'confidence': min(1.0, confidence),
                    'energy_level': energy_smooth[idx]
                })
        
        return pause_locations
    
    def _apply_pause_enhancements(self, audio: np.ndarray, pause_locations: List[Dict[str, Any]], 
                                 source_profile: Dict[str, Any]) -> np.ndarray:
        """
        Apply gentle pause enhancements at specified locations
        """
        if not pause_locations:
            return audio
        
        # Sort by position
        pause_locations.sort(key=lambda x: x['position'])
        
        enhanced_segments = []
        last_position = 0
        
        for pause_info in pause_locations:
            position = pause_info['position']
            confidence = pause_info['confidence']
            
            # Add audio up to this position
            start_sample = int(last_position * self.sample_rate)
            end_sample = int(position * self.sample_rate)
            
            if end_sample > start_sample and end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                enhanced_segments.append(segment)
                
                # Add a gentle pause extension based on confidence and source characteristics
                if confidence > 0.5:
                    pause_extension = min(0.3, source_profile['average_pause'] * confidence * 0.5)
                    pause_samples = int(pause_extension * self.sample_rate)
                    
                    if pause_samples > 0:
                        # Create natural pause with room tone
                        natural_pause = self._create_natural_pause(pause_samples, audio, start_sample, end_sample)
                        enhanced_segments.append(natural_pause)
                
                last_position = position
        
        # Add remaining audio
        if last_position < len(audio) / self.sample_rate:
            remaining_start = int(last_position * self.sample_rate)
            if remaining_start < len(audio):
                enhanced_segments.append(audio[remaining_start:])
        
        return np.concatenate(enhanced_segments) if enhanced_segments else audio
    
    def _create_natural_pause(self, pause_samples: int, context_audio: np.ndarray, 
                             start_sample: int, end_sample: int) -> np.ndarray:
        """
        Create a natural pause with room tone characteristics
        """
        # Get context for room tone
        context_start = max(0, start_sample - 1000)
        context_end = min(len(context_audio), end_sample + 1000)
        context = context_audio[context_start:context_end]
        
        if len(context) > 0:
            # Use the quietest part of the context as room tone
            frame_size = min(512, len(context) // 4)
            if frame_size > 0:
                context_frames = librosa.util.frame(context, frame_length=frame_size, hop_length=frame_size//2)
                frame_energies = np.mean(context_frames ** 2, axis=0)
                quietest_frame_idx = np.argmin(frame_energies)
                
                start_idx = quietest_frame_idx * (frame_size // 2)
                end_idx = start_idx + frame_size
                
                if end_idx <= len(context):
                    room_tone = context[start_idx:end_idx]
                    
                    # Extend room tone to desired length
                    if len(room_tone) > 0:
                        repeat_count = (pause_samples // len(room_tone)) + 1
                        extended_room_tone = np.tile(room_tone, repeat_count)[:pause_samples]
                        
                        # Apply gentle fade to avoid clicks
                        fade_samples = min(pause_samples // 20, int(0.01 * self.sample_rate))
                        if fade_samples > 0:
                            fade_in = np.linspace(0, 1, fade_samples)
                            fade_out = np.linspace(1, 0, fade_samples)
                            extended_room_tone[:fade_samples] *= fade_in
                            extended_room_tone[-fade_samples:] *= fade_out
                        
                        # Reduce volume significantly
                        return extended_room_tone * 0.1
        
        # Fallback to very quiet noise
        return np.random.normal(0, 0.001, pause_samples).astype(np.float32)