from ultralytics import YOLO
import logging
import torch

logger = logging.getLogger(__name__)

class LeopardDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config.get('device') == 'cuda' else 'cpu'
        logger.info(f"Loading YOLOv8 model from {model_path} on {self.device}...")
        try:
            self.model = YOLO(model_path)
            # Warmup with dummy image
            import numpy as np
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(source=dummy_frame, device=self.device, verbose=False)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, frame):
        """
        Run inference on a frame.
        """
        conf_thres = self.config.get('conf_threshold', 0.6)
        iou_thres = self.config.get('iou_threshold', 0.45)
        classes = self.config.get('classes', [0]) # Default class filter
        
        results = self.model.predict(
            frame, 
            conf=conf_thres, 
            iou=iou_thres, 
            classes=classes, 
            verbose=False
        )
        return results[0]  # Return first result (single frame)
