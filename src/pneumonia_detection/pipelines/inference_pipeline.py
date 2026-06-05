import sys
from pathlib import Path

# Ensure model packages are visible in system path BEFORE imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

import argparse
import json
import logging

from src.pneumonia_detection.utils.config import load_all_configs
from src.pneumonia_detection.utils.logging import configure_logger
from src.pneumonia_detection.inference.predict import Predictor

logger = logging.getLogger(__name__)


def run_inference_pipeline() -> None:
    """Execute batch or single-image inference based on configuration and arguments."""
    parser = argparse.ArgumentParser(description="Pneumonia Detection Inference Pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        default=None, 
        help="Path to a single X-Ray image or a directory containing multiple X-Ray images."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/inference.yaml", 
        help="Path to inference configuration YAML file or config directory."
    )
    args = parser.parse_args()

    try:
        # 1. Load Configurations
        config_path = Path(args.config)
        if config_path.is_file():
            config = load_all_configs(config_path.parent)
        else:
            config = load_all_configs(config_path)
        
        # 2. Setup Central Logger
        log_cfg = config["training"]["logging"]
        configure_logger(
            log_dir=log_cfg.get("dir", "logs"),
            log_file="inference.log",
            level=log_cfg.get("level", "INFO")
        )
        
        logger.info("Initializing Inference Pipeline...")
        
        # 3. Instantiate Predictor Engine
        checkpoint_path = Path(config["inference"]["checkpoint_path"])
        if not checkpoint_path.exists():
            logger.error(
                f"Model checkpoint not found at: {checkpoint_path.resolve()}. "
                "Please run the training pipeline first or configure inference.yaml with a valid checkpoint path."
            )
            sys.exit(1)
            
        predictor = Predictor(config)
        
        # 4. Resolve Input Target
        input_target = args.input if args.input is not None else config["data"]["image_dir"]
        target_path = Path(input_target)
        
        if not target_path.exists():
            logger.error(f"Provided input path does not exist: {target_path.resolve()}")
            sys.exit(1)
            
        # 5. Output configurations
        output_dir = Path(config["inference"].get("output_dir", "predictions"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. Execute predictions
        if target_path.is_file():
            logger.info(f"Executing single image prediction on: {target_path.name}")
            prob, class_idx, label = predictor.predict_image(target_path)
            
            result = {
                "file_path": str(target_path),
                "file_name": target_path.name,
                "probability": prob,
                "class_idx": class_idx,
                "label": label
            }
            
            logger.info(f"Result: {label} (Confidence: {prob * 100.0:.2f}%)")
            
            # Save prediction
            save_path = output_dir / f"prediction_{target_path.stem}.json"
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Prediction result saved to: {save_path}")
            
        elif target_path.is_dir():
            logger.info(f"Executing batch folder predictions on: {target_path.resolve()}")
            results = predictor.predict_batch(target_path)
            
            if len(results) == 0:
                logger.warning("No predictions were generated.")
                return
                
            # Save predictions list
            save_path = output_dir / "batch_predictions.json"
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Batch prediction results saved to: {save_path}")
            
            # Print a summary of counts
            predictions_labels = [r["label"] for r in results]
            normal_cnt = predictions_labels.count("NORMAL")
            pneu_cnt = predictions_labels.count("PNEUMONIA")
            logger.info(f"Batch prediction summary - Normal: {normal_cnt} | Pneumonia: {pneu_cnt}")
            
    except Exception as e:
        logger.exception(f"Unhandled exception during inference pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference_pipeline()
