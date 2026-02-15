import albumentations as A
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmentor:
    def __init__(self, input_dir="dataset/processed", output_dir="dataset/augmented", num_augmentations=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_augmentations = num_augmentations
        
        # Robustness-focused augmentation pipeline
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            # Weather simulation
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=7, p=1),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, p=1),
                A.RandomShadow(p=1),
            ], p=0.3),
        ])

    def augment_dataset(self):
        logger.info("Starting data augmentation...")
        
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            output_class_dir = self.output_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
            
            logger.info(f"Augmenting class {class_dir.name} with {len(images)} source images...")
            
            for img_path in tqdm(images):
                try:
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Copy original
                    cv2.imwrite(str(output_class_dir / img_path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Generate augmentations
                    for i in range(self.num_augmentations):
                        augmented = self.transform(image=image)['image']
                        aug_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        cv2.imwrite(str(output_class_dir / aug_filename), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                        
                except Exception as e:
                    logger.error(f"Error augmenting {img_path}: {e}")
                    
        logger.info("Data augmentation complete.")

if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.augment_dataset()
