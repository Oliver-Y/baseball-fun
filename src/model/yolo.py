"""
Convert Parquet data to YOLO format and train YOLO models.

YOLO format:
- One .txt file per image with annotations
- Format: class_id center_x center_y width height (all normalized 0-1)
- Directory structure:
  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/
"""
import os
#import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
#import numpy as np
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import pandas as pd
#from ultralytics import YOLO

from src.model.db import read_from_parquet

logger = logging.getLogger(__name__)

# YOLO class ID for baseball (single class)
BASEBALL_CLASS_ID = 0


def coco_to_yolo(bbox_x: float, bbox_y: float, bbox_width: float, bbox_height: float,
                 image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO format bbox (x_min, y_min, width, height) to YOLO format (center_x, center_y, width, height).
    
    Args:
        bbox_x: Top-left x coordinate (pixels)
        bbox_y: Top-left y coordinate (pixels)
        bbox_width: Width (pixels)
        bbox_height: Height (pixels)
        image_width: Image width (pixels)
        image_height: Image height (pixels)
        
    Returns:
        Tuple of (center_x, center_y, width, height) normalized to [0, 1]
    """
    # Calculate center coordinates
    center_x = (bbox_x + bbox_width / 2) / image_width
    center_y = (bbox_y + bbox_height / 2) / image_height
    
    # Normalize width and height
    norm_width = bbox_width / image_width
    norm_height = bbox_height / image_height
    
    # Clamp to [0, 1] range
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return center_x, center_y, norm_width, norm_height


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        # Fallback: try with cv2
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
        raise


def convert_to_yolo_format(data: dict, output_labels_dir: str) -> str:
    """
    Convert a single annotation entry to YOLO format.
    
    Args:
        data: Dictionary with keys: image_path, bbox_x, bbox_y, bbox_width, bbox_height
        output_labels_dir: Directory to save label files
        
    Returns:
        Path to the created label file
    """
    # Get image dimensions
    image_width, image_height = get_image_dimensions(data['image_path'])
    
    # Convert bbox to YOLO format
    center_x, center_y, norm_width, norm_height = coco_to_yolo(
        data['bbox_x'], data['bbox_y'], data['bbox_width'], data['bbox_height'],
        image_width, image_height
    )
    
    # Create label file path (same name as image but with .txt extension)
    image_name = Path(data['image_path']).stem
    label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
    
    # Write YOLO format annotation
    # Format: class_id center_x center_y width height
    with open(label_path, 'w') as f:
        f.write(f"{BASEBALL_CLASS_ID} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    return label_path

def create_yolo_directory_structure(output_dir: str) -> Dict[str, Tuple[Path, Path]]:
    output_path = Path(output_dir)
    images_train_dir = output_path / "images" / "train"
    images_val_dir = output_path / "images" / "val"
    labels_train_dir = output_path / "labels" / "train"
    labels_val_dir = output_path / "labels" / "val"
    images_test_dir = output_path / "images" / "test"
    labels_test_dir = output_path / "labels" / "test"
    
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    
    images_test_dir.mkdir(parents=True, exist_ok=True)
    labels_test_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (images_train_dir, labels_train_dir),
        "val": (images_val_dir, labels_val_dir),
        "test": (images_test_dir, labels_test_dir),
    }


def create_yolo_dataset(parquet_path: str, output_dir: str, train_split: float = 0.8, 
                       val_split: float = 0.2, test_split: float = 0.0):
    """
    Convert Parquet data to YOLO dataset format.
    
    Args:
        parquet_path: Path to input Parquet file
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (default 0.8)
        val_split: Fraction of data for validation (default 0.2)
        test_split: Fraction of data for testing (default 0.0)
    """
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Read data from Parquet (use DataFrame for efficiency)
    logger.info(f"Reading data from {parquet_path}...")
    df = read_from_parquet(parquet_path, format="dataframe")
    logger.info(f"Loaded {len(df)} annotations")
    
    # Create YOLO directory structure
    directories = create_yolo_directory_structure(output_dir)
    
    ## Split data
    logger.info(f"Splitting data: train={train_split:.1%}, val={val_split:.1%}, test={test_split:.1%}")
    if test_split > 0:
        train_val_data, test_data = train_test_split(df, test_size=test_split, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=val_split/(train_split+val_split), random_state=42)
    else:
        train_data, val_data = train_test_split(df, test_size=val_split, random_state=42)
        test_data = []
    
    
    logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Process each split, format is (data_list, directory_tuple, split_name)
    splits = [
        (train_data, directories["train"], "train"),
        (val_data, directories["val"], "val"),
    ]
    if test_split > 0:
        splits.append((test_data, directories["test"], "test"))
    
    # TODO: Potentially vectorize but for now we can just go row by row
    for data_list, dirs, split_name in splits:
        images_dir, labels_dir = dirs
        logger.info(f"Processing {split_name} split: {len(data_list)} samples")
        
        for idx, entry in enumerate(data_list.itertuples()):
            dest_image_path = images_dir / Path(entry.image_path).name
            
            # Create symlink instead of copying (saves disk space)
            if not dest_image_path.exists():
                src_path = Path(entry.image_path).resolve()
                os.symlink(src_path, dest_image_path)
            
            # Create YOLO label file
            convert_to_yolo_format(entry._asdict(), str(labels_dir))
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(data_list)} {split_name} samples")
        
        logger.info(f"✓ Completed {split_name} split: {len(data_list)} samples")
    
    # Create dataset.yaml file for YOLO
    output_path = Path(output_dir)
    create_yolo_config(output_path, num_classes=1, class_names=["baseball"])
    
    logger.info(f"✓ YOLO dataset created at {output_dir}")

def create_yolo_config(dataset_dir: Path, num_classes: int = 1, class_names: List[str] = None):
    """
    Create YOLO dataset configuration file (dataset.yaml).
    
    Args:
        dataset_dir: Dataset root directory
        num_classes: Number of classes
        class_names: List of class names
    """
    if class_names is None:
        class_names = ["baseball"]
    
    # Get absolute paths
    dataset_path = dataset_dir.absolute()
    
    config_content = f"""# YOLO Dataset Configuration
# Path to dataset (relative to this file or absolute)

path: {dataset_path}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
# test: images/test  # optional test images (relative to 'path')

# Classes
nc: {num_classes}  # number of classes
names: {class_names}  # class names
"""
    
    config_path = dataset_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✓ Created YOLO config at {config_path}")


def train_yolo_model(dataset_yaml: str, model_size: str = "n", epochs: int = 100, 
                    imgsz: int = 640, batch: int = 16, device: str = "0"):
    """
    Train a YOLO model using ultralytics.
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
    """

    logger.info(f"Initializing YOLOv8{model_size} model...")
    model = YOLO(f"yolov8{model_size}.pt")  # Load pretrained model
    
    logger.info(f"Starting training on {dataset_yaml}...")
    logger.info(f"  Model: YOLOv8{model_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Device: {device}")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="baseball_detection",
        exist_ok=True,
    )
    
    logger.info("✓ Training completed!")
    logger.info(f"  Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Convert Parquet to YOLO format and train model")
    parser.add_argument("--parquet", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split (default: 0.2)")
    parser.add_argument("--test-split", type=float, default=0.0, help="Test split (default: 0.0)")
    parser.add_argument("--train", action="store_true", help="Train YOLO model after conversion")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="YOLO model size (default: n)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device (default: 0 for GPU, 'cpu' for CPU)")
    
    args = parser.parse_args()
    
    # Convert to YOLO format
    create_yolo_dataset(
        args.parquet,
        args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Train model if requested
    #if args.train:
    #    dataset_yaml = os.path.join(args.output, "dataset.yaml")
    #    train_yolo_model(
    #        dataset_yaml,
    #        model_size=args.model_size,
    #        epochs=args.epochs,
    #        batch=args.batch,
    #        imgsz=args.imgsz,
    #        device=args.device
    #    )

