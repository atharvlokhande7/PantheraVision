import logging
import threading
import queue
import time
import cv2
import sqlite3
from pathlib import Path
from datetime import datetime
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.alert_queue = queue.Queue()
        self.setup_db()
        self.running = True
        self.thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.thread.start()

    def setup_db(self):
        db_path = self.config.get('database', {}).get('path', 'data.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                confidence REAL,
                image_path TEXT,
                video_path TEXT,
                metadata TEXT
            )
        ''')
        self.conn.commit()

    def trigger_alert(self, detection_data, frame):
        """
        Queue an alert.
        detection_data: dict containing confidence, bbox, etc.
        """
        self.alert_queue.put({'data': detection_data, 'frame': frame})

    def _process_alerts(self):
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._handle_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    def _handle_alert(self, alert):
        data = alert['data']
        frame = alert['frame']
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save Image
        img_dir = Path(self.config.get('system', {}).get('output_dir', 'output')) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"leopard_{timestamp}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Log to DB
        self.cursor.execute('''
            INSERT INTO detections (timestamp, confidence, image_path, video_path, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), data['conf'], str(img_path), "", json.dumps(data)))
        self.conn.commit()
        
        logger.info(f"Leopard Detected! Conf: {data['conf']:.2f}. Saved to {img_path}")
        
        # Send Telegram
        if self.config.get('alerts', {}).get('telegram', {}).get('enabled'):
            self._send_telegram(str(img_path), f"🐆 Leopard Detected! Conf: {data['conf']:.2f}")

    def _send_telegram(self, image_path, caption):
        try:
            tg_conf = self.config['alerts']['telegram']
            token = tg_conf['token']
            chat_id = tg_conf['chat_id']
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            
            with open(image_path, 'rb') as f:
                payload = {'chat_id': chat_id, 'caption': caption}
                files = {'photo': f}
                requests.post(url, data=payload, files=files, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def stop(self):
        self.running = False
        self.thread.join()
        self.conn.close()
