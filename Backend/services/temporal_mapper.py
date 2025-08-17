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

from typing import Optional

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
        Enhanced energy analysis with better silence detection
        """
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Multi-feature analysis
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        
        # Spectral features for better voice activity detection
        stft = librosa.stft(audio, hop_length=hop_length, n_fft=1024)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=self.sample_rate)[0]
        
        # Smooth all features
        energy_smooth = gaussian_filter1d(energy, sigma=3.0)
        centroids_smooth = gaussian_filter1d(spectral_centroids, sigma=2.0)
        rolloff_smooth = gaussian_filter1d(spectral_rolloff, sigma=2.0)
        
        # Enhanced voice activity detection using multiple features
        energy_threshold = np.percentile(energy_smooth, 25)
        centroid_threshold = np.percentile(centroids_smooth, 30)
        rolloff_threshold = np.percentile(rolloff_smooth, 30)
        
        # Combine features for more accurate VAD
        voice_confidence = (
            (energy_smooth > energy_threshold).astype(float) * 0.5 +
            (centroids_smooth > centroid_threshold).astype(float) * 0.3 +
            (rolloff_smooth > rolloff_threshold).astype(float) * 0.2
        )
        
        # More conservative voice activity detection
        voiced_frames = voice_confidence > 0.6
        
        # Find speech onset and offset more accurately
        speech_onset_frame = self._find_speech_onset(voiced_frames)
        speech_offset_frame = self._find_speech_offset(voiced_frames)
        
        # Calculate timing with onset/offset awareness
        total_frames = len(voiced_frames)
        onset_time = speech_onset_frame * hop_length / self.sample_rate if speech_onset_frame else 0
        offset_time = speech_offset_frame * hop_length / self.sample_rate if speech_offset_frame else len(audio) / self.sample_rate
        
        # More accurate pause detection
        pause_segments = self._detect_pause_segments(voiced_frames, hop_length)
        
        duration = len(audio) / self.sample_rate
        speaking_ratio = np.mean(voiced_frames)
        
        return {
            'total_duration': duration,
            'speaking_time': duration * speaking_ratio,
            'pause_time': sum(p['duration'] for p in pause_segments),
            'speaking_ratio': speaking_ratio,
            'average_pause': np.mean([p['duration'] for p in pause_segments]) if pause_segments else 0.3,
            'num_pauses': len(pause_segments),
            'speech_rate': speaking_ratio * 3.0,
            'onset_time': onset_time,
            'offset_time': offset_time,
            'pause_segments': pause_segments
        }
    def _find_speech_onset(self, voiced_frames: np.ndarray) -> Optional[int]:
        """
        Find the first significant speech onset
        """
        if len(voiced_frames) == 0:
            return None
        
        # Look for sustained speech (not just a blip)
        min_speech_duration = int(0.3 * self.sample_rate / (0.010 * self.sample_rate))  # 300ms in frames
        
        for i in range(len(voiced_frames) - min_speech_duration):
            # Check if we have sustained speech
            if np.mean(voiced_frames[i:i + min_speech_duration]) > 0.7:
                return i
        
        # Fallback to first voiced frame
        voiced_indices = np.where(voiced_frames)[0]
        return voiced_indices[0] if len(voiced_indices) > 0 else None

    def _find_speech_offset(self, voiced_frames: np.ndarray) -> Optional[int]:
        """
        Find the last significant speech offset
        """
        if len(voiced_frames) == 0:
            return None
        
        # Look backwards for sustained speech
        min_speech_duration = int(0.3 * self.sample_rate / (0.010 * self.sample_rate))  # 300ms in frames
        
        for i in range(len(voiced_frames) - 1, min_speech_duration, -1):
            # Check if we have sustained speech ending here
            if np.mean(voiced_frames[i - min_speech_duration:i]) > 0.7:
                return i
        
        # Fallback to last voiced frame
        voiced_indices = np.where(voiced_frames)[0]
        return voiced_indices[-1] if len(voiced_indices) > 0 else None

    def _detect_pause_segments(self, voiced_frames: np.ndarray, hop_length: int) -> List[Dict[str, float]]:
        """
        Detect meaningful pause segments with better accuracy
        """
        if len(voiced_frames) == 0:
            return []
        
        # Find unvoiced regions
        unvoiced_regions = ~voiced_frames
        
        # Find starts and ends of unvoiced regions
        diff = np.diff(unvoiced_regions.astype(int))
        pause_starts = np.where(diff == 1)[0] + 1
        pause_ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if unvoiced_regions[0]:
            pause_starts = np.insert(pause_starts, 0, 0)
        if unvoiced_regions[-1]:
            pause_ends = np.append(pause_ends, len(unvoiced_regions))
        
        pause_segments = []
        for start, end in zip(pause_starts, pause_ends):
            duration = (end - start) * hop_length / self.sample_rate
            
            # Only consider significant pauses
            if duration > self.min_significant_pause:
                pause_segments.append({
                    'start_time': start * hop_length / self.sample_rate,
                    'end_time': end * hop_length / self.sample_rate,
                    'duration': duration,
                    'position': (start + end) / 2 * hop_length / self.sample_rate
                })
        
        return pause_segments
    
    def apply_temporal_guidance(self, target_audio: np.ndarray, source_profile: Dict[str, Any]) -> np.ndarray:
        """
        Apply enhanced temporal guidance with better onset/offset awareness
        """
        target_duration = len(target_audio) / self.sample_rate
        
        # Analyze target audio with enhanced detection
        target_profile = self._energy_based_profile(target_audio)
        
        source_speaking_ratio = source_profile['speaking_ratio']
        target_speaking_ratio = target_profile['speaking_ratio']
        
        # Check if we need onset/offset adjustments
        needs_onset_adjustment = abs(source_profile.get('onset_time', 0) - target_profile.get('onset_time', 0)) > 0.2
        needs_pause_adjustment = abs(source_speaking_ratio - target_speaking_ratio) > 0.15
        
        if needs_onset_adjustment or needs_pause_adjustment:
            logger.info(f"Applying enhanced temporal guidance: source_onset={source_profile.get('onset_time', 0):.2f}s, "
                    f"target_onset={target_profile.get('onset_time', 0):.2f}s, "
                    f"source_ratio={source_speaking_ratio:.2f}, target_ratio={target_speaking_ratio:.2f}")
            
            # Apply onset adjustment first
            onset_adjusted_audio = self._apply_onset_adjustment(target_audio, source_profile, target_profile)
            
            # Then apply gentle stretching if needed
            if abs(source_speaking_ratio - target_speaking_ratio) > 0.2:
                tempo_adjusted_audio = self._apply_gentle_stretching(onset_adjusted_audio, source_profile)
            else:
                tempo_adjusted_audio = onset_adjusted_audio
            
            # Finally apply enhanced pause adjustments
            final_audio = self._apply_enhanced_pause_adjustments(tempo_adjusted_audio, source_profile)
            
            return final_audio
        else:
            logger.info("Target audio timing already well-matched, applying minimal enhancement")
            return self._apply_enhanced_pause_adjustments(target_audio, source_profile)
        
    def _apply_onset_adjustment(self, audio: np.ndarray, source_profile: Dict[str, Any], 
                           target_profile: Dict[str, Any]) -> np.ndarray:
        """
        Adjust speech onset timing to better match source characteristics
        """
        source_onset = source_profile.get('onset_time', 0)
        target_onset = target_profile.get('onset_time', 0)
        
        onset_difference = source_onset - target_onset
        
        # Only adjust if difference is significant and not too extreme
        if abs(onset_difference) > 0.1 and abs(onset_difference) < 2.0:
            if onset_difference > 0:
                # Source starts later, add some initial silence/room tone
                silence_samples = int(min(onset_difference, 1.0) * self.sample_rate)
                
                # Create natural-sounding initial pause
                if len(audio) > 1000:
                    # Use beginning of audio as room tone template
                    room_tone_template = audio[:500]
                    room_tone_level = np.std(room_tone_template) * 0.3
                    initial_silence = np.random.normal(0, room_tone_level, silence_samples).astype(np.float32)
                else:
                    initial_silence = np.zeros(silence_samples, dtype=np.float32)
                
                return np.concatenate([initial_silence, audio])
            else:
                # Source starts earlier, trim some initial audio (be conservative)
                trim_samples = int(min(abs(onset_difference) * 0.5, 0.5) * self.sample_rate)
                if trim_samples < len(audio):
                    return audio[trim_samples:]
        
        return audio
    
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
    
    def _apply_enhanced_pause_adjustments(self, audio: np.ndarray, source_profile: Dict[str, Any]) -> np.ndarray:
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