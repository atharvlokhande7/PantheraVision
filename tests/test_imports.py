import sys
import os
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports...")
    try:
        from app.camera import CameraStream
        print("CameraStream imported")
        from app.streaming import StreamServer
        print("StreamServer imported")
        from motion.optical_flow import MotionDetector
        print("MotionDetector imported")
        from detection.model import LeopardDetector
        print("LeopardDetector imported")
        from detection.filters import DetectionFilter
        print("DetectionFilter imported")
        from tracking.tracker import ObjectTracker
        print("ObjectTracker imported")
        from alerts.notifier import AlertSystem
        print("AlertSystem imported")
        from app.pipeline import Pipeline
        print("Pipeline imported")
        print("All imports successful!")
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
