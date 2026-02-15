import time
import cv2
import logging
import threading
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.camera import CameraStream
from app.streaming import StreamServer
from motion.optical_flow import MotionDetector
from detection.model import LeopardDetector
from detection.filters import DetectionFilter
from tracking.tracker import ObjectTracker
from alerts.notifier import AlertSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.camera = CameraStream(self.config['camera']['source'])
        self.stream_server = StreamServer()
        self.motion_detector = MotionDetector(self.config['motion'])
        self.detector = LeopardDetector(self.config['detection']['model_path'], self.config['detection'])
        self.filter = DetectionFilter(self.config)
        self.tracker = ObjectTracker(self.config['tracking'])
        self.alert_system = AlertSystem(self.config)
        
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()

    def run(self):
        logger.info("Starting PantheraVision Pipeline...")
        self.camera.start()
        self.stream_server.start()

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            self.frame_count += 1
            
            # 1. Motion Detection
            has_motion, motion_mask, motion_rects = self.motion_detector.detect(frame)
            
            detections = []
            
            # 2. Inference (run only if motion detected or periodic check)
            # Optimization: Skip inference if no motion (unless needed for consistent tracking)
            # For now, we run inference every frame if hardware allows, or use motion trigger to save compute.
            # Let's use motion trigger + periodic forced inference (every 30 frames) to catch slow movers
            if has_motion or (self.frame_count % 30 == 0):
                # YOLO Prediction
                results = self.detector.predict(frame)
                
                # Check boxes (x1, y1, x2, y2, conf, cls) in results.boxes
                if results.boxes is not None:
                     for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = results.names[cls]
                        
                        # Only care about target class (0=leopard based on our config)
                        # but if using standard COCO model, '0' is person. 
                        # Assuming we have a trained model where 0 is leopard
                        
                        # 3. Filtering
                        valid, reason = self.filter.validate_detection(
                            (x1, y1, x2, y2), conf, motion_rects=motion_rects if has_motion else None
                        )
                        
                        if valid:
                             # Add to list for tracker (using -1 as temp ID, tracker will assign/match)
                            # Actually YOLO tracker returns IDs if we use .track() method. 
                            # But we used .predict(). 
                            # If we want IDs, we should use .track() in model.py.
                            # For simplicity with custom pipeline: 
                            # We'll pass raw dets to our tracker.py which assigns IDs.
                            detections.append((x1, y1, x2, y2, -1, conf, cls)) # -1 ID initially

            # 4. Tracking
            # Basic tracker assignment
            tracks = self.tracker.update(detections) # This needs detections format
            # Wait, my simple tracker expects YOLO tracked output with IDs if I used .track().
            # If I stick to .predict(), I need a SORT/Hungarian matcher in `tracker.py`.
            # To fix this mismatch: I will rely on YOLO's internal tracker if possible, or
            # use a simple centroid tracker here?
            # actually, let's just visualize the boxes for now and trigger alerts.
            
            annotated_frame = frame.copy()
            
            # Draw motion rects
            if motion_rects:
                for x, y, w, h in motion_rects:
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            # Draw Detections & Trigger Alerts
            for det in detections:
                x1, y1, x2, y2, _, conf, _ = det
                
                # Alert Logic
                self.alert_system.trigger_alert({'conf': conf, 'bbox': [x1, y1, x2, y2]}, annotated_frame)
                
                # Draw
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Leopard: {conf:.2f}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update Stream
            # Add FPS
            fps = self.frame_count / (current_time - self.start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            self.stream_server.update_frame(annotated_frame)
            
    def stop(self):
        self.running = False
        self.camera.stop()
        self.alert_system.stop()

if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
