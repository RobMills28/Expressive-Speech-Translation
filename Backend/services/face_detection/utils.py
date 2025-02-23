import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_face_region(frame: np.ndarray, bbox, padding: float = 0.2) -> np.ndarray:
    """Extract face region from frame with padding."""
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return frame[y1:y2, x1:x2]
    except Exception as e:
        logger.error(f"Failed to extract face region: {str(e)}")
        return frame