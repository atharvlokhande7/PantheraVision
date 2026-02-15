import os
import requests
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataScraper:
    def __init__(self, download_dir="dataset/raw", max_workers=8):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def download_image(self, url, class_name):
        try:
            class_dir = self.download_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename from hash of URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            file_path = class_dir / f"{url_hash}.jpg"

            if file_path.exists():
                return  # Skip existing

            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                # Basic validity check using OpenCV
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if img is not None:
                    # Save locally
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    logger.warning(f"Invalid image found at {url}")
            else:
                logger.warning(f"Failed to download {url}: Status {response.status_code}")

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")

    def download_batch(self, urls, class_name):
        logger.info(f"Downloading {len(urls)} images for class '{class_name}'...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(lambda url: self.download_image(url, class_name), urls), total=len(urls)))

    def download_kaggle_dataset(self, dataset_name, path=None):
        """
        Download a dataset from Kaggle.
        Requires KAGGLE_USERNAME and KAGGLE_KEY env vars or ~/.kaggle/kaggle.json
        """
        try:
            import kaggle
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            kaggle.api.dataset_download_files(dataset_name, path=path or self.download_dir, unzip=True)
            logger.info("Download complete.")
        except ImportError:
            logger.error("Kaggle API not installed. Please install with `pip install kaggle`.")
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")

if __name__ == "__main__":
    # Example usage
    scraper = DataScraper()
    
    # Placeholder for actual URL lists
    leopard_urls = [
        # "https://example.com/leopard1.jpg",
        # "https://example.com/leopard2.jpg"
    ]
    
    # scraper.download_batch(leopard_urls, "leopard")
    # scraper.download_kaggle_dataset("soumikrakshit/animal-image-dataset-90-different-animals")
