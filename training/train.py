import argparse
import logging
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_best_model_variant():
    """
    Auto-select YOLOv8 variant based on available GPU memory.
    """
    if not torch.cuda.is_available():
        logger.warning("No GPU detected. Defaulting to YOLOv8n (nano) for CPU training.")
        return "yolov8n.pt"

    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        logger.info(f"Detected GPU with {gpu_mem:.2f} GB VRAM.")

        if gpu_mem > 20:
            return "yolov8x.pt"  # Extra Large
        elif gpu_mem > 12:
            return "yolov8l.pt"  # Large
        elif gpu_mem > 8:
            return "yolov8m.pt"  # Medium
        else:
            return "yolov8s.pt"  # Small
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}. Defaulting to YOLOv8s.")
        return "yolov8s.pt"

def train_model(data_yaml, epochs=100, img_size=640, batch_size=16, model_variant=None):
    if model_variant is None:
        model_variant = select_best_model_variant()
    
    logger.info(f"Starting training with model: {model_variant}")
    logger.info(f"Data config: {data_yaml}")
    
    # Load model
    model = YOLO(model_variant)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True,
        save_period=5,
        patience=20,  # Early stopping
        device=0 if torch.cuda.is_available() else 'cpu',
        project="models",
        name="leopard_detector",
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        rect=True, # Rectangular training for faster training
        cos_lr=True, # Cosine learning rate scheduler
        close_mosaic=10, # Disable mosaic augmentation for final 10 epochs
        warmup_epochs=3.0,
        box=7.5, # Box loss gain
        cls=0.5, # Class loss gain
        dfl=1.5, # DFL loss gain
    )
    
    logger.info("Training complete.")
    logger.info(f"Best model saved at: {results.save_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Leopard Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default=None, help="YOLOv8 model variant (n/s/m/l/x)")
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.img_size, args.batch, args.model)
