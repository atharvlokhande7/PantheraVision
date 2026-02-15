from flask import Flask, Response, render_template_string, jsonify
import threading
import cv2
import time
import logging

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
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

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
                img { border: 2px solid #00ff00; max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>🐆 PantheraVision Live Stream</h1>
            <img src="/video">
            <p>Status: <span style="color: #00ff00;">Active</span></p>
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
