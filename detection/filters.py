import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

class DetectionFilter:
    def __init__(self, config):
        self.config = config
        self.history = {} # Track ID -> history of detections
        self.min_hits = config.get('tracking', {}).get('min_hits', 3)
        self.min_confidence = config.get('detection', {}).get('conf_threshold', 0.6)

    def validate_detection(self, bbox, confidence, track_id=None, motion_rects=None):
        """
        Validates a detection based on heuristic rules and history.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # 1. Aspect Ratio Check (Leopards are generally horizontal)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 4.0:
            return False, "Invalid aspect ratio"
            
        # 2. Minimum Size Check
        if w < 50 or h < 50:
             return False, "Too small"

        # 3. Motion Validation (Intersection with motion mask)
        if motion_rects:
            has_overlap = False
            box_area = w * h
            
            for mx, my, mw, mh in motion_rects:
                # Calculate Intersection
                ix1 = max(x1, mx)
                iy1 = max(y1, my)
                ix2 = min(x2, mx + mw)
                iy2 = min(y2, my + mh)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                intersection = iw * ih
                if intersection > 0.3 * box_area: # at least 30% overlap with motion
                    has_overlap = True
                    break
            
            if not has_overlap:
                return False, "No motion correlation"

        # 4. Temporal Consistency (if tracking enabled)
        if track_id is not None:
            if track_id not in self.history:
                self.history[track_id] = {'hits': 0, 'first_seen': time.time()}
            
            self.history[track_id]['hits'] += 1
            
            if self.history[track_id]['hits'] < self.min_hits:
                return False, f"Waiting for confirmation ({self.history[track_id]['hits']}/{self.min_hits})"

        return True, "Valid"

    def clean_history(self, current_track_ids):
        """Cleanup old tracks"""
        for tid in list(self.history.keys()):
            if tid not in current_track_ids:
                del self.history[tid]
