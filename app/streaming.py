from flask import Flask, Response, render_template_string, jsonify
import threading
import cv2
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global frame buffer for streaming
outputFrame = None
lock = threading.Lock()

def set_output_frame(frame):
    global outputFrame
    with lock:
        outputFrame = frame.copy()

def generate():
    global outputFrame
    
    # Pre-create blank frame to disable re-creation overhead
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Status: Waiting for Camera...", (50, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(blank_frame, "No input source detected on Server", (50, 270), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    while True:
        frame_to_encode = None
        
        with lock:
            if outputFrame is None:
                frame_to_encode = blank_frame
            else:
                frame_to_encode = outputFrame
        
        # Encode
        try:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag:
                time.sleep(0.1)
                continue
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
        
        # Control FPS
        time.sleep(0.05)

@app.route("/video")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template_string("""
    <html>
        <head>
            <title>PantheraVision - Leopard Detection System</title>
            <style>
                body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; }
                img { border: 2px solid #00ff00; max-width: 100%; height: auto; background-color: #000; }
                .info { margin-top: 20px; color: #aaa; }
            </style>
        </head>
        <body>
            <h1>🐆 PantheraVision Live Stream</h1>
            <img src="/video" alt="Live Stream">
            <div class="info">
                <p>Status: <span style="color: #00ff00;">Active</span></p>
                <p><small>Note: This system runs on the Server. It does not access your browser webcam.</small></p>
            </div>
        </body>
    </html>
    """)

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

class StreamServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        # Disable Flask banner
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)

    def update_frame(self, frame):
        set_output_frame(frame)
