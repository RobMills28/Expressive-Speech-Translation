# services/visual_temporal_mapper.py (COMPLETE FIXED VERSION)
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from scipy.ndimage import gaussian_filter1d

from .visual_speech_detector import VisualSpeechDetector

logger = logging.getLogger(__name__)

class VisualTemporalMapper:
    """
    Fixed temporal mapping that properly handles both single and multiple speech segments
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.visual_detector = VisualSpeechDetector()
        
        # Simple parameters for natural audio flow
        self.fade_duration = 0.02  # 20ms fade in/out to avoid clicks
        self.natural_pause_ratio = 0.15  # 15% of speech time can be pauses
        
    def initialize(self):
        """Initialize the visual speech detector"""
        return self.visual_detector.initialize()
    
    def apply_visual_temporal_mapping(self, translated_audio: np.ndarray, 
                                     original_video_path: Path,
                                     process_id_short: str) -> np.ndarray:
        """
        Apply natural temporal mapping that properly distributes audio across speech periods
        """
        logger.info(f"[{process_id_short}] Applying natural temporal mapping with intelligent audio distribution")
        
        try:
            # Analyze original video for speech activity pattern
            visual_analysis = self.visual_detector.analyze_video_speech_activity(original_video_path)
            
            # Get the speech segments and video info
            speech_segments = visual_analysis['speech_segments']
            total_video_duration = visual_analysis['total_duration']
            speech_ratio = visual_analysis['speech_ratio']
            
            logger.info(f"[{process_id_short}] Video analysis: {len(speech_segments)} speech segments, "
                       f"total duration: {total_video_duration:.2f}s, speech ratio: {speech_ratio:.2f}")
            
            # Handle different scenarios based on detected segments
            if len(speech_segments) == 0:
                # No speech detected - just place audio naturally with some leading pause
                synchronized_audio = self._add_minimal_leading_pause(translated_audio, process_id_short)
                
            elif len(speech_segments) == 1:
                # Single large segment - need to intelligently distribute audio within it
                synchronized_audio = self._distribute_in_single_segment(
                    translated_audio, speech_segments[0], total_video_duration, process_id_short
                )
                
            else:
                # Multiple segments - distribute across them naturally
                synchronized_audio = self._distribute_across_multiple_segments(
                    translated_audio, speech_segments, total_video_duration, process_id_short
                )
            
            return synchronized_audio
            
        except Exception as e:
            logger.warning(f"[{process_id_short}] Visual mapping failed: {e}, returning original audio")
            return translated_audio
    
    def _distribute_in_single_segment(self, audio: np.ndarray, 
                                     speech_segment: Dict[str, float],
                                     video_duration: float,
                                     process_id_short: str) -> np.ndarray:
        """
        Handle the case where we detect one large speech segment - distribute audio intelligently within it
        """
        audio_duration = len(audio) / self.sample_rate
        segment_start = speech_segment['start']
        segment_duration = speech_segment['duration']
        
        logger.info(f"[{process_id_short}] Single segment detected ({segment_duration:.2f}s), "
                   f"distributing {audio_duration:.2f}s of audio intelligently within it")
        
        # Create output buffer that's long enough for natural speech flow
        output_duration = max(audio_duration * 1.2, video_duration)  # 20% buffer for natural flow
        output_samples = int(output_duration * self.sample_rate)
        output_audio = np.zeros(output_samples, dtype=np.float32)
        
        # Split audio into natural chunks based on energy patterns
        audio_chunks = self._split_audio_by_energy_patterns(audio, min_chunks=3, max_chunks=8)
        
        logger.info(f"[{process_id_short}] Split audio into {len(audio_chunks)} natural chunks")
        
        # Distribute chunks within the speech segment with natural spacing
        final_output = self._place_chunks_with_natural_spacing_fixed(
            audio_chunks, output_audio, segment_start, segment_duration, process_id_short
        )
        
        return final_output
    
    def _distribute_across_multiple_segments(self, audio: np.ndarray, 
                                           speech_segments: List[Dict[str, float]],
                                           video_duration: float,
                                           process_id_short: str) -> np.ndarray:
        """
        Distribute audio across multiple detected speech segments
        """
        audio_duration = len(audio) / self.sample_rate
        
        logger.info(f"[{process_id_short}] Multiple segments detected, distributing "
                   f"{audio_duration:.2f}s across {len(speech_segments)} segments")
        
        # Create output buffer
        output_duration = max(audio_duration * 1.1, video_duration)
        output_samples = int(output_duration * self.sample_rate)
        output_audio = np.zeros(output_samples, dtype=np.float32)
        
        # Split audio to match number of segments (roughly)
        audio_chunks = self._split_audio_by_energy_patterns(audio, 
                                                           min_chunks=len(speech_segments), 
                                                           max_chunks=len(speech_segments) * 2)
        
        # Place chunks in segments
        for i, (chunk, segment) in enumerate(zip(audio_chunks, speech_segments)):
            if i >= len(speech_segments):
                break
                
            # Place chunk within this segment with small natural offset
            segment_start = segment['start']
            natural_offset = min(0.3, segment['duration'] * 0.2)  # Up to 20% into segment
            placement_time = segment_start + natural_offset
            
            start_sample = int(placement_time * self.sample_rate)
            end_sample = start_sample + len(chunk)
            
            if end_sample <= len(output_audio):
                faded_chunk = self._apply_gentle_fade(chunk)
                output_audio[start_sample:start_sample + len(faded_chunk)] = faded_chunk
                
                logger.debug(f"[{process_id_short}] Placed chunk {i+1} at {placement_time:.2f}s "
                           f"(duration: {len(chunk)/self.sample_rate:.2f}s)")
        
        return output_audio
    
    def _place_chunks_with_natural_spacing(self, audio_chunks: List[np.ndarray],
                                          output_audio: np.ndarray,
                                          segment_start: float,
                                          segment_duration: float,
                                          process_id_short: str):
        """
        FIXED: Place audio chunks within a single segment with natural spacing and pauses
        """
        if not audio_chunks:
            return
        
        # Calculate total audio duration
        total_audio_duration = sum(len(chunk) / self.sample_rate for chunk in audio_chunks)
        
        logger.info(f"[{process_id_short}] FIXED PLACEMENT: {len(audio_chunks)} chunks, "
                   f"total audio: {total_audio_duration:.2f}s, "
                   f"segment: {segment_start:.2f}s-{segment_start + segment_duration:.2f}s "
                   f"({segment_duration:.2f}s available)")
        
        # Calculate available space and natural distribution
        available_time = segment_duration
        
        # Don't compress audio too much - if it doesn't fit naturally, extend the output
        if total_audio_duration > available_time * 0.9:  # If audio takes more than 90% of available time
            # Extend output buffer to accommodate natural speech flow
            extended_duration = total_audio_duration * 1.3  # 30% buffer for natural pauses
            needed_samples = int((segment_start + extended_duration) * self.sample_rate)
            
            if needed_samples > len(output_audio):
                # Extend the output buffer
                additional_samples = needed_samples - len(output_audio)
                extension = np.zeros(additional_samples, dtype=np.float32)
                output_audio = np.concatenate([output_audio, extension])
                
                logger.info(f"[{process_id_short}] Extended output buffer by {additional_samples/self.sample_rate:.2f}s "
                           f"to accommodate natural speech flow")
        
        # Calculate natural pause distribution
        if len(audio_chunks) > 1:
            # Distribute audio chunks naturally across the available space
            total_space_needed = total_audio_duration * 1.2  # 20% buffer for pauses
            space_per_chunk = total_space_needed / len(audio_chunks)
            
            # Calculate pause between chunks
            pause_per_gap = (space_per_chunk - (total_audio_duration / len(audio_chunks))) * 0.8
            pause_per_gap = max(0.1, min(pause_per_gap, 0.5))  # Keep pauses reasonable (0.1s - 0.5s)
        else:
            pause_per_gap = 0
        
        logger.info(f"[{process_id_short}] Placement strategy: {pause_per_gap:.2f}s pause between chunks")
        
        # Place chunks with calculated spacing
        current_time = segment_start + 0.1  # Small initial offset
        chunks_placed = 0
        
        for i, chunk in enumerate(audio_chunks):
            chunk_duration = len(chunk) / self.sample_rate
            start_sample = int(current_time * self.sample_rate)
            end_sample = start_sample + len(chunk)
            
            # Check if we need to extend output buffer further
            if end_sample > len(output_audio):
                additional_samples = end_sample - len(output_audio) + int(0.5 * self.sample_rate)  # Extra 0.5s buffer
                extension = np.zeros(additional_samples, dtype=np.float32)
                
                # If output_audio is a view/slice, we need to create a new array
                try:
                    output_audio = np.concatenate([output_audio, extension])
                except ValueError:
                    # If we can't extend, place what we can
                    chunk = chunk[:len(output_audio) - start_sample]
                    end_sample = len(output_audio)
                    if len(chunk) <= 0:
                        logger.warning(f"[{process_id_short}] Chunk {i+1} completely truncated - buffer too small")
                        continue
                
                logger.info(f"[{process_id_short}] Extended buffer during placement by {additional_samples/self.sample_rate:.2f}s")
            
            if start_sample < len(output_audio) and len(chunk) > 0:
                # Apply fade and place chunk
                faded_chunk = self._apply_gentle_fade(chunk)
                
                # Ensure we don't go out of bounds
                actual_end = min(start_sample + len(faded_chunk), len(output_audio))
                actual_chunk = faded_chunk[:actual_end - start_sample]
                
                if len(actual_chunk) > 0:
                    output_audio[start_sample:start_sample + len(actual_chunk)] = actual_chunk
                    chunks_placed += 1
                    
                    logger.info(f"[{process_id_short}] ✓ Placed chunk {i+1}/{len(audio_chunks)} "
                               f"at {current_time:.2f}s (duration: {chunk_duration:.2f}s)")
                    
                    # Move to next position with natural pause
                    current_time += chunk_duration + pause_per_gap
                else:
                    logger.warning(f"[{process_id_short}] ✗ Failed to place chunk {i+1} - no space")
            else:
                logger.warning(f"[{process_id_short}] ✗ Skipped chunk {i+1} - start position {start_sample} invalid")
        
        logger.info(f"[{process_id_short}] PLACEMENT COMPLETE: {chunks_placed}/{len(audio_chunks)} chunks placed successfully")
        
        # Final verification - check that we actually have audio content
        final_content_check = np.max(np.abs(output_audio))
        logger.info(f"[{process_id_short}] Final audio peak amplitude: {final_content_check:.4f}")
        
        if final_content_check < 0.001:
            logger.error(f"[{process_id_short}] WARNING: Final output appears to be silent!")
    
    def _distribute_in_single_segment(self, audio: np.ndarray, 
                                     speech_segment: Dict[str, float],
                                     video_duration: float,
                                     process_id_short: str) -> np.ndarray:
        """
        FIXED: Handle the case where we detect one large speech segment - distribute audio intelligently within it
        """
        audio_duration = len(audio) / self.sample_rate
        segment_start = speech_segment['start']
        segment_duration = speech_segment['duration']
        
        logger.info(f"[{process_id_short}] FIXED Single segment detected ({segment_duration:.2f}s), "
                   f"distributing {audio_duration:.2f}s of audio intelligently within it")
        
        # Create output buffer that's guaranteed to be large enough
        # Use the longer of: natural audio length + buffer, or video duration + buffer
        min_output_duration = max(
            audio_duration * 1.5,  # 50% buffer for natural flow
            video_duration,        # At least as long as video
            segment_start + segment_duration + 1.0  # Segment plus 1s buffer
        )
        
        output_samples = int(min_output_duration * self.sample_rate)
        output_audio = np.zeros(output_samples, dtype=np.float32)
        
        logger.info(f"[{process_id_short}] Created output buffer: {min_output_duration:.2f}s ({output_samples} samples)")
        
        # Split audio into natural chunks based on energy patterns
        audio_chunks = self._split_audio_by_energy_patterns(audio, min_chunks=3, max_chunks=8)
        
        logger.info(f"[{process_id_short}] Split audio into {len(audio_chunks)} natural chunks")
        
        # Log chunk details
        for i, chunk in enumerate(audio_chunks):
            chunk_duration = len(chunk) / self.sample_rate
            chunk_rms = np.sqrt(np.mean(chunk ** 2))
            logger.debug(f"[{process_id_short}] Chunk {i+1}: {chunk_duration:.2f}s, RMS: {chunk_rms:.4f}")
        
        # CRITICAL FIX: Pass the output_audio array properly and handle extension
        self._place_chunks_with_natural_spacing_fixed(
            audio_chunks, output_audio, segment_start, segment_duration, process_id_short
        )
        
        return output_audio
    
    def _place_chunks_with_natural_spacing_fixed(self, audio_chunks: List[np.ndarray],
                                                output_audio: np.ndarray,
                                                segment_start: float,
                                                segment_duration: float,
                                                process_id_short: str) -> np.ndarray:
        """
        COMPLETELY FIXED version that ensures all audio is placed and handles buffer extension
        """
        if not audio_chunks:
            return output_audio
        
        # Calculate total audio duration
        total_audio_duration = sum(len(chunk) / self.sample_rate for chunk in audio_chunks)
        
        logger.info(f"[{process_id_short}] FIXED PLACEMENT v2: {len(audio_chunks)} chunks, "
                   f"total audio: {total_audio_duration:.2f}s, "
                   f"segment: {segment_start:.2f}s-{segment_start + segment_duration:.2f}s")
        
        # Strategy: Place ALL audio chunks sequentially with natural pauses
        # Don't try to fit within the detected segment if it's too constraining
        
        # Calculate spacing
        if len(audio_chunks) > 1:
            # Natural pause between chunks (0.2-0.4 seconds)
            pause_per_gap = min(0.4, max(0.2, total_audio_duration * 0.1 / len(audio_chunks)))
        else:
            pause_per_gap = 0
        
        total_time_needed = total_audio_duration + (len(audio_chunks) - 1) * pause_per_gap
        
        logger.info(f"[{process_id_short}] Placement plan: {pause_per_gap:.2f}s between chunks, "
                   f"{total_time_needed:.2f}s total needed")
        
        # Start placement with small offset from segment start
        current_time = segment_start + 0.2
        chunks_placed = 0
        placed_chunks_info = []
        
        for i, chunk in enumerate(audio_chunks):
            chunk_duration = len(chunk) / self.sample_rate
            start_sample = int(current_time * self.sample_rate)
            end_sample = start_sample + len(chunk)
            
            # Extend output buffer if needed
            while end_sample > len(output_audio):
                # Add 2 seconds worth of samples
                extension_samples = int(2.0 * self.sample_rate)
                extension = np.zeros(extension_samples, dtype=np.float32)
                output_audio = np.concatenate([output_audio, extension])
                
                logger.info(f"[{process_id_short}] Extended output buffer to {len(output_audio)/self.sample_rate:.2f}s")
            
            # Place the chunk
            if start_sample >= 0 and start_sample < len(output_audio):
                faded_chunk = self._apply_gentle_fade(chunk)
                
                # Ensure we don't exceed bounds
                actual_end = min(start_sample + len(faded_chunk), len(output_audio))
                chunk_to_place = faded_chunk[:actual_end - start_sample]
                
                if len(chunk_to_place) > 0:
                    output_audio[start_sample:start_sample + len(chunk_to_place)] = chunk_to_place
                    chunks_placed += 1
                    
                    placed_info = {
                        'chunk_index': i + 1,
                        'start_time': current_time,
                        'duration': len(chunk_to_place) / self.sample_rate,
                        'samples_placed': len(chunk_to_place)
                    }
                    placed_chunks_info.append(placed_info)
                    
                    logger.info(f"[{process_id_short}] ✓ PLACED chunk {i+1}/{len(audio_chunks)} "
                               f"at {current_time:.2f}s (duration: {chunk_duration:.2f}s)")
                    
                    # Move to next position
                    current_time += chunk_duration + pause_per_gap
                else:
                    logger.error(f"[{process_id_short}] ✗ FAILED to place chunk {i+1} - zero length after bounds check")
            else:
                logger.error(f"[{process_id_short}] ✗ FAILED to place chunk {i+1} - invalid start position {start_sample}")
        
        # Final summary
        logger.info(f"[{process_id_short}] PLACEMENT SUMMARY:")
        logger.info(f"  Chunks processed: {len(audio_chunks)}")
        logger.info(f"  Chunks placed successfully: {chunks_placed}")
        logger.info(f"  Output buffer size: {len(output_audio)/self.sample_rate:.2f}s")
        
        # Verify we have audio content
        final_peak = np.max(np.abs(output_audio))
        final_rms = np.sqrt(np.mean(output_audio ** 2))
        
        logger.info(f"  Final audio peak: {final_peak:.4f}")
        logger.info(f"  Final audio RMS: {final_rms:.4f}")
        
        if final_peak < 0.001:
            logger.error(f"[{process_id_short}] CRITICAL: Output audio appears to be silent!")
        else:
            logger.info(f"[{process_id_short}] ✓ Output audio has content")
        
        return output_audio
    
    def _split_audio_by_energy_patterns(self, audio: np.ndarray, 
                                       min_chunks: int = 2, 
                                       max_chunks: int = 8) -> List[np.ndarray]:
        """
        Split audio into natural chunks based on energy patterns and pauses
        """
        if len(audio) == 0:
            return []
        
        # Analyze audio energy to find natural split points
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Calculate energy
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        energy_smooth = gaussian_filter1d(energy, sigma=3.0)
        
        # Find low-energy regions (potential split points)
        energy_threshold = np.percentile(energy_smooth, 20)  # Lower 20% energy
        low_energy_frames = energy_smooth < energy_threshold
        
        # Find transitions from high to low energy (potential pause starts)
        energy_changes = np.diff(low_energy_frames.astype(int))
        pause_starts = np.where(energy_changes == 1)[0]  # Start of low energy
        
        # Convert frame indices to time
        potential_split_times = pause_starts * hop_length / self.sample_rate
        
        # Filter splits that are too close together (minimum 0.5s apart)
        filtered_splits = []
        last_split = -1.0
        for split_time in potential_split_times:
            if split_time - last_split > 0.5:
                filtered_splits.append(split_time)
                last_split = split_time
        
        # Limit number of chunks
        if len(filtered_splits) > max_chunks - 1:
            # Keep the most significant splits (those with lowest energy)
            split_energies = []
            for split_time in filtered_splits:
                frame_idx = int(split_time * self.sample_rate / hop_length)
                if frame_idx < len(energy_smooth):
                    split_energies.append((split_time, energy_smooth[frame_idx]))
            
            # Sort by energy and take the lowest
            split_energies.sort(key=lambda x: x[1])
            filtered_splits = [s[0] for s in split_energies[:max_chunks-1]]
            filtered_splits.sort()
        
        # Ensure minimum number of chunks
        audio_duration = len(audio) / self.sample_rate
        if len(filtered_splits) < min_chunks - 1:
            # Add evenly spaced splits
            needed_splits = min_chunks - 1 - len(filtered_splits)
            for i in range(needed_splits):
                split_time = (i + 1) * audio_duration / (needed_splits + 1)
                filtered_splits.append(split_time)
            filtered_splits.sort()
        
        # Convert to sample indices and create chunks
        split_samples = [int(t * self.sample_rate) for t in filtered_splits]
        split_samples = [0] + split_samples + [len(audio)]
        
        chunks = []
        for i in range(len(split_samples) - 1):
            start_idx = split_samples[i]
            end_idx = split_samples[i + 1]
            if end_idx > start_idx:  # Ensure non-empty chunks
                chunks.append(audio[start_idx:end_idx])
        
        return chunks
    
    def _add_minimal_leading_pause(self, audio: np.ndarray, process_id_short: str) -> np.ndarray:
        """
        Add minimal leading pause when no speech segments detected
        """
        # Add small natural pause (0.2-0.5s)
        pause_duration = 0.3
        pause_samples = int(pause_duration * self.sample_rate)
        
        # Create subtle room tone
        room_tone = self._create_room_tone(pause_samples, audio)
        
        logger.info(f"[{process_id_short}] No speech segments detected, adding {pause_duration}s leading pause")
        
        return np.concatenate([room_tone, audio])
    
    def _apply_gentle_fade(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gentle fade in/out to avoid audio clicks
        """
        if len(audio) == 0:
            return audio
        
        fade_samples = int(self.fade_duration * self.sample_rate)
        fade_samples = min(fade_samples, len(audio) // 4)  # Don't fade more than 25% of audio
        
        if fade_samples > 0:
            faded_audio = audio.copy()
            
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            faded_audio[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            faded_audio[-fade_samples:] *= fade_out
            
            return faded_audio
        
        return audio
    
    def _create_room_tone(self, samples: int, reference_audio: np.ndarray) -> np.ndarray:
        """
        Create natural room tone based on the quietest part of reference audio
        """
        if len(reference_audio) == 0 or samples <= 0:
            return np.zeros(samples, dtype=np.float32)
        
        # Find the quietest section of the reference audio
        window_size = min(int(0.5 * self.sample_rate), len(reference_audio) // 4)
        if window_size <= 0:
            # Fallback to very quiet noise
            return np.random.normal(0, 0.001, samples).astype(np.float32)
        
        # Calculate RMS energy for sliding windows
        min_energy = float('inf')
        quietest_start = 0
        
        for i in range(0, len(reference_audio) - window_size, window_size // 4):
            window = reference_audio[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            if energy < min_energy:
                min_energy = energy
                quietest_start = i
        
        # Extract the quietest section
        quietest_section = reference_audio[quietest_start:quietest_start + window_size]
        
        # Reduce amplitude and repeat to desired length
        room_tone_template = quietest_section * 0.1  # Much quieter
        
        # Tile to desired length
        repeat_count = (samples // len(room_tone_template)) + 1
        room_tone = np.tile(room_tone_template, repeat_count)[:samples]
        
        return room_tone.astype(np.float32)
    
    def cleanup(self):
        """Clean up resources"""
        self.visual_detector.cleanup()
        logger.debug("Visual temporal mapper cleaned up")