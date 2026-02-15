# Troubleshooting Guide

## Common Issues

### 1. Stream "Keeps Loading" or is Black
- **Cause**: The server cannot connect to a camera.
- **Solution**: 
  - Ensure your camera is connected to the **EC2 Instance / Server** (not your laptop).
  - Verify the RTSP URL in `configs/config.yaml`.
  - The stream now shows a status message "Waiting for Camera..." if no source is found.

### 2. "It doesn't show camera permission"
- **Cause**: This is a **Server-Side** detection system. It processes video on the server, not in your browser.
- **Explanation**: 
  - Your browser is effectively a TV screen receiving a broadcast.
  - The camera must be plugged into the **Broadcast Station (EC2 Server)**.
  - Therefore, your browser does not need (and will not ask for) camera permissions.
- **Fix**: Connect a USB camera to the server or use an IP Camera URL.

### 3. "Connection Refused"
- **Cause**: The server is not running or crashed.
- **Fix**:
  ```bash
  cd infrastructure
  docker-compose logs -f
  ```
  Check for errors. If it crashed, restart with `docker-compose up -d`.

### 4. GPU Errors
- **Cause**: Running on a machine without NVIDIA GPU.
- **Fix**: The system automatically falls back to CPU, but ensure `docker-compose.yml` does NOT have the `deploy.resources.reservations` block enabled (it is commented out by default).
