import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class ObjectTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = {}  # track_id -> TrackState
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        
    def update(self, detections):
        """
        Update tracks with new detections from YOLO tracker.
        detections: List of (x1, y1, x2, y2, id, conf, class)
        """
        current_ids = set()
        
        for det in detections:
            if len(det) < 7: # Standard YOLO format with track ID
                continue
                
            x1, y1, x2, y2, track_id, conf, cls = det
            track_id = int(track_id)
            current_ids.add(track_id)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'hits': 1,
                    'last_seen': time.time(),
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'status': 'probation',
                    'history': deque(maxlen=30)
                }
            else:
                track = self.tracks[track_id]
                track['hits'] += 1
                track['last_seen'] = time.time()
                track['bbox'] = (x1, y1, x2, y2)
                track['conf'] = max(track['conf'], conf)
                track['history'].append((x1, y1, x2, y2))
                
                if track['hits'] >= self.min_hits:
                    track['status'] = 'confirmed'

        # Cleanup lost tracks
        self._prune_tracks(current_ids)
        
        return self.tracks

    def _prune_tracks(self, current_ids):
        now = time.time()
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                age = now - self.tracks[track_id]['last_seen']
                if age > self.max_age: # Remove if not seen for max_age seconds? No, usually max_age frames.
                    # Adapting to time-based for robustness against FPS drops
                    del self.tracks[track_id]
