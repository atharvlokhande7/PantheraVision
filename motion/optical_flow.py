import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, config):
        self.config = config
        self.prev_gray = None
        self.hsv = None
        self.motion_mask = None

    def detect(self, frame):
        """
        Detects motion in the frame using Farneback Optical Flow.
        Returns a tuple (has_motion, motion_mask, bounding_rects)
        """
        if frame is None:
            return False, None, []

        # Resize for performance improvement (optional, but recommended for optical flow)
        small_frame = cv2.resize(frame, (640, 480)) 
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.hsv = np.zeros_like(small_frame)
            self.hsv[..., 1] = 255
            return False, None, []

        # Calculate Optical Flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold to create motion mask
        # Sensitivity controlled by var_threshold in config (mapped to magnitude threshold here)
        threshold = self.config.get('var_threshold', 2.0)
        motion_mask = mag > threshold
        
        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = np.uint8(motion_mask) * 255
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)

        # Find contours of motion
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_rects = []
        has_motion = False
        min_area = self.config.get('min_area', 500)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            
            has_motion = True
            # Build bounding rect (scaled back to original size)
            x, y, w, h = cv2.boundingRect(contour)
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            motion_rects.append((
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y)
            ))

        self.prev_gray = gray
        
        return has_motion, motion_mask, motion_rects
