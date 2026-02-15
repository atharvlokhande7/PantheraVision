import cv2
import time
import logging
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, source, reconnect_interval=5, buffer_size=128):
        self.source = source
        self.reconnect_interval = reconnect_interval
        self.frame_queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Determine if source is int (webcam) or str (file/rtsp)
        try:
            self.source = int(self.source)
        except ValueError:
            pass

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        cap = cv2.VideoCapture(self.source)
        
        while not self.stopped:
            if not cap.isOpened():
                logger.warning(f"Camera {self.source} disconnected. Reconnecting in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(self.source)
                continue

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {self.source}. Reconnecting...")
                cap.release()
                time.sleep(self.reconnect_interval)
                cap = cv2.VideoCapture(self.source)
                continue

            # Keep queue full (drop oldest if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put(frame)

        cap.release()

    def read(self):
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
