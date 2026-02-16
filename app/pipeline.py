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

        # Check if source is a file
        source = self.config['camera']['source']
        self.is_file = isinstance(source, str) and Path(source).exists()
        
        self.camera = CameraStream(source, file_mode=self.is_file)
        self.stream_server = StreamServer()
        self.motion_detector = MotionDetector(self.config['motion'])
        self.detector = LeopardDetector(self.config['detection']['model_path'], self.config['detection'])
        self.filter = DetectionFilter(self.config)
        self.tracker = ObjectTracker(self.config['tracking'])
        self.alert_system = AlertSystem(self.config)
        
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # specific to output video
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / "leopard_detection_output.mp4"
        self.video_writer = None

    def run(self):
        logger.info("Starting PantheraVision Pipeline...")
        self.camera.start()
        self.stream_server.start()

        while self.running:
            frame = self.camera.read()
            if frame is None:
                if self.is_file:
                    logger.info("End of video file reached.")
                    break
                time.sleep(0.01)
                continue

            current_time = time.time()
            self.frame_count += 1
            
            # 1. Motion Detection
            has_motion, motion_mask, motion_rects = self.motion_detector.detect(frame)
            
            detections = []
            
            # 2. Inference (run every frame if file mode to assure accuracy, or skip if needed)
            # For video file output, we generally want every frame processed for smoothness
            if self.is_file or has_motion or (self.frame_count % 30 == 0):
                # YOLO Prediction
                results = self.detector.predict(frame)
                
                # Check boxes (x1, y1, x2, y2, conf, cls) in results.boxes
                if results.boxes is not None:
                     for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = results.names[cls]
                        
                        # UX Improvement: Map 'cat' to 'Leopard'
                        if class_name.lower() == 'cat':
                            class_name = 'Leopard'
                        
                        # 3. Filtering
                        valid, reason = self.filter.validate_detection(
                            (x1, y1, x2, y2), conf, motion_rects=motion_rects if has_motion else None
                        )
                        
                        if valid:
                            detections.append((x1, y1, x2, y2, -1, conf, cls)) # -1 ID initially

            # 4. Tracking
            tracks = self.tracker.update(detections)
            
            annotated_frame = frame.copy()
            
            # Draw motion rects
            if motion_rects:
                for x, y, w, h in motion_rects:
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            # Draw Detections & Trigger Alerts
            if detections:
                # Flashing Alert
                if int(time.time() * 5) % 2 == 0: # Flash every 0.2s
                    cv2.putText(annotated_frame, "WARNING: LEOPARD DETECTED!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
            for det in detections:
                x1, y1, x2, y2, _, conf, _ = det
                
                # Alert Logic
                self.alert_system.trigger_alert({'conf': conf, 'bbox': [x1, y1, x2, y2]}, annotated_frame)
                
                # Draw Box ONLY (No Text)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Update Stream
            self.stream_server.update_frame(annotated_frame)
            
            # Write to video file
            if self.video_writer is None:
                height, width = annotated_frame.shape[:2]
                logger.info(f"Initializing VideoWriter with {width}x{height} @ 30.0 fps")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_path = self.output_path.with_suffix('.mp4') # Ensure mp4 extension
                self.video_writer = cv2.VideoWriter(str(self.output_path), fourcc, 30.0, (width, height))
                
                if not self.video_writer.isOpened():
                    logger.error("Failed to open VideoWriter!")
                else:
                    logger.info(f"VideoWriter initialized successfully at {self.output_path}")
            
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.write(annotated_frame)
            
    def stop(self):
        self.running = False
        self.camera.stop()
        self.alert_system.stop()
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"Output video saved to {self.output_path}")

if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
