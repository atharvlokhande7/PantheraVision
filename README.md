# PantheraVision: Leopard Detection System

A production-grade, real-time Leopard Detection System using YOLOv8 and Optical Flow, designed for AWS EC2 deployment with live streaming capabilities.

## Features

*   **Real-time Detection**: Uses YOLOv8 for high-accuracy object detection.
*   **Motion Filtering**: Optical flow (Farneback) integration to reduce false positives from static backgrounds.
*   **Live Streaming**: Low-latency MJPEG streaming via Flask.
*   **Robustness**: Handles camera reconnects, lighting changes, and weather simulation augmentation.
*   **Alerts**: Telegram integration and local database logging.
*   **Deployment**: Dockerized for easy deployment on AWS EC2 (GPU supported).

## Project Structure

```
leopard-detector/
├── app/                 # Application logic & streaming
├── motion/              # Optical flow & motion detection
├── detection/           # YOLOv8 inference & filtering
├── tracking/            # Object tracking
├── alerts/              # Notification system
├── dataset/             # Data tools (scraper, cleaner, augment)
├── training/            # Training scripts
├── configs/             # Configuration files
├── infrastructure/      # Docker & systemd files
└── requirements.txt
```

## Quick Start (Local)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure**:
    Edit `configs/config.yaml` to set your camera source (uses webcam '0' by default).

3.  **Run**:
    ```bash
    python app/main.py
    ```

4.  **View Stream**:
    Open `http://localhost:5000/video` in your browser.

## Deployment (AWS EC2)

1.  **Launch EC2**: Use an instance with GPU (e.g., `g4dn.xlarge`) running Ubuntu.
2.  **Clone Repository**:
    ```bash
    git clone <your-repo-url>
    cd leopard-detector
    ```
3.  **Run with Docker**:
    ```bash
    cd infrastructure
    docker-compose up --build -d
    ```
4.  **Access**:
    Ensure port 5000 is open in your Security Group. Access via `http://<EC2-IP>:5000/video`.

## Training

Refer to `training/README.md` (to be created) for details on training the custom YOLOv8 model.

## License

MIT