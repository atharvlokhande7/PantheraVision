from ultralytics import YOLO
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_path, data_yaml):
    logger.info(f"Evaluating model: {model_path}")
    model = YOLO(model_path)
    
    # Validate on test/val set
    metrics = model.val(data=data_yaml, split='val', verbose=True, save_json=True, plots=True)
    
    logger.info(f"mAP@0.5: {metrics.box.map50}")
    logger.info(f"mAP@0.5:0.95: {metrics.box.map}")
    logger.info(f"Precision: {metrics.box.mp}")
    logger.info(f"Recall: {metrics.box.mr}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Model")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt model")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data)
