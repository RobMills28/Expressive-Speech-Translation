import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe face detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame and return bounding boxes.
        Returns: List of (x1, y1, x2, y2) coordinates
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Get detections
            results = self.face_detection.process(rgb_frame)
            faces = []
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    faces.append((x1, y1, x2, y2))
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []

    def get_main_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest/most central face in the frame."""
        faces = self.detect_faces(frame)
        if not faces:
            return None
            
        # Get the largest face by area
        areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in faces]
        return faces[areas.index(max(areas))]