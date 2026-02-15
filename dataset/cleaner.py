import cv2
import hashlib
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, raw_dir="dataset/raw", processed_dir="dataset/processed", min_res=(640, 640), blur_threshold=100.0):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.min_res = min_res
        self.blur_threshold = blur_threshold
        self.hashes = set()

    def get_image_hash(self, image):
        # Convert to grayscale and resize for hashing
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        # Binary string based on average
        diff = gray > avg
        # Convert binary string to hex
        return hashlib.md5(diff.tobytes()).hexdigest()

    def is_blurry(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < self.blur_threshold

    def process_dataset(self):
        logger.info("Starting dataset cleaning...")
        
        # Traverse all class directories
        for class_dir in self.raw_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            output_class_dir = self.processed_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing class: {class_dir.name}")
            
            images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
            
            for img_path in tqdm(images):
                try:
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        logger.warning(f"Corrupt image removed: {img_path}")
                        continue
                        
                    # 1. Minimum Resolution Check
                    h, w = img.shape[:2]
                    if w < self.min_res[0] or h < self.min_res[1]:
                        # logger.info(f"Skipping low res ({w}x{h}): {img_path}")
                        continue
                        
                    # 2. Blur Check
                    if self.is_blurry(img):
                        # logger.info(f"Skipping blurry image: {img_path}")
                        continue
                        
                    # 3. Duplicate Check
                    img_hash = self.get_image_hash(img)
                    if img_hash in self.hashes:
                        # logger.info(f"Skipping duplicate: {img_path}")
                        continue
                    
                    self.hashes.add(img_hash)
                    
                    # Save valid image
                    shutil.copy2(img_path, output_class_dir / img_path.name)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")

        logger.info("Dataset cleaning complete.")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process_dataset()
