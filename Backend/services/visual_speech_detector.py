# services/visual_speech_detector.py (SIMPLIFIED VERSION)
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class VisualSpeechDetector:
    """
    Simplified speech detector focused on basic mouth movement detection for MVP
    """
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        # Simplified parameters for MVP
        self.movement_threshold = 0.005  # Lower threshold for broader detection
        self.min_speech_duration = 0.5   # Longer minimum for stability
        self.frame_skip = 3               # Process every 3rd frame for speed
        
        # Key mouth landmarks (simplified set)
        self.mouth_landmarks = [
            61, 84, 17, 314, 405, 320, 307, 375,  # Outer mouth
            12, 15, 16, 18, 200, 199, 175         # Inner mouth
        ]
        
    def initialize(self):
        """Initialize MediaPipe with error handling"""
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,  # Faster processing
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3
            )
            logger.info("Visual speech detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize visual speech detector: {e}")
            return False
    
    def analyze_video_speech_activity(self, video_path: Path) -> Dict[str, Any]:
        """
        Simplified video analysis focused on basic speech detection
        """
        logger.info(f"Analyzing speech activity in: {video_path.name}")
        
        if not self.face_mesh:
            if not self.initialize():
                raise RuntimeError("Failed to initialize face mesh")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get basic video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
        
        # Track mouth movement
        mouth_movements = []
        frame_times = []
        frame_count = 0
        processed_frames = 0
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Skip frames for faster processing
            if frame_count % (self.frame_skip + 1) == 0:
                frame_times.append(timestamp)
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    # Calculate mouth movement
                    movement = self._calculate_simple_mouth_movement(
                        results.multi_face_landmarks[0], frame.shape
                    )
                    mouth_movements.append(movement)
                else:
                    mouth_movements.append(0.0)
                
                processed_frames += 1
                
                # Log progress periodically
                if processed_frames % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.debug(f"Processed {processed_frames} frames ({progress:.1f}%)")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Processed {processed_frames} frames from {total_frames} total")
        
        # Detect speech segments from movement data
        speech_segments = self._detect_simple_speech_segments(mouth_movements, frame_times)
        
        # Calculate basic statistics
        total_speech_time = sum(seg['duration'] for seg in speech_segments)
        speech_ratio = total_speech_time / duration if duration > 0 else 0
        
        logger.info(f"Detected {len(speech_segments)} speech segments, "
                   f"speech ratio: {speech_ratio:.2f}")
        
        return {
            'speech_segments': speech_segments,
            'total_duration': duration,
            'fps': fps,
            'speech_ratio': speech_ratio,
            'processed_frames': processed_frames
        }
    
    def _calculate_simple_mouth_movement(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> float:
        """
        Calculate basic mouth movement - simplified for MVP
        """
        height, width = frame_shape[:2]
        
        try:
            # Get mouth landmark coordinates
            mouth_points = []
            for idx in self.mouth_landmarks:
                landmark = face_landmarks.landmark[idx]
                x = landmark.x * width
                y = landmark.y * height
                mouth_points.append([x, y])
            
            mouth_points = np.array(mouth_points)
            
            # Calculate mouth area as a simple movement indicator
            # Use convex hull for robustness
            hull = cv2.convexHull(mouth_points.astype(np.int32))
            area = cv2.contourArea(hull)
            
            # Normalize by face size (approximate)
            face_size = width * height * 0.1  # Rough face size estimate
            normalized_area = area / face_size if face_size > 0 else 0
            
            return min(1.0, normalized_area)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Error calculating mouth movement: {e}")
            return 0.0
    
    def _detect_simple_speech_segments(self, movements: List[float], 
                                      timestamps: List[float]) -> List[Dict[str, float]]:
        """
        Simple speech segment detection based on movement threshold
        """
        if not movements or not timestamps:
            return []
        
        # Apply simple threshold
        is_speaking = [movement > self.movement_threshold for movement in movements]
        
        # Find speech segments
        segments = []
        in_speech = False
        speech_start = None
        
        for i, speaking in enumerate(is_speaking):
            timestamp = timestamps[i]
            
            if speaking and not in_speech:
                # Start of speech
                speech_start = timestamp
                in_speech = True
            elif not speaking and in_speech:
                # End of speech
                if speech_start is not None:
                    duration = timestamp - speech_start
                    if duration >= self.min_speech_duration:
                        segments.append({
                            'start': speech_start,
                            'end': timestamp,
                            'duration': duration,
                            'type': 'speech'
                        })
                in_speech = False
                speech_start = None
        
        # Handle speech continuing to end
        if in_speech and speech_start is not None:
            final_time = timestamps[-1] if timestamps else 0
            duration = final_time - speech_start
            if duration >= self.min_speech_duration:
                segments.append({
                    'start': speech_start,
                    'end': final_time,
                    'duration': duration,
                    'type': 'speech'
                })
        
        # Merge close segments
        merged_segments = self._merge_close_segments(segments)
        
        return merged_segments
    
    def _merge_close_segments(self, segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Merge segments that are very close together
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            
            # Merge if gap is less than 0.5 seconds
            if gap < 0.5:
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged
    
    def create_audio_sync_template(self, speech_segments: List[Dict[str, float]], 
                                  total_duration: float) -> Dict[str, Any]:
        """
        Create a simple sync template for audio placement
        """
        total_speech_time = sum(seg['duration'] for seg in speech_segments)
        speech_ratio = total_speech_time / total_duration if total_duration > 0 else 0.8
        
        return {
            'total_duration': total_duration,
            'speech_segments': speech_segments,
            'speech_ratio': speech_ratio,
            'num_segments': len(speech_segments)
        }
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        logger.debug("Visual speech detector cleaned up")