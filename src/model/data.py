# Have this module pull down data in batches, generate the trajectories, move it to an image, 
# annotate the image, and store in different data formats
#Data formats: 
# 1. COCO format
# 2. YOLO format 
# What is the goal: Real-time inference, on-device for model within 0.5s latency
from typing import List, Dict, Iterator
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from src import utils
import logging
from datetime import datetime
from src.traj_generation.trajectory_viz import generate_trajectory_points
from src.traj_generation.frame_angle import calculate_ball_data, generate_video_frames, calculate_2d_points
from src.virtual_camera import VirtualCamera
from src.model.db import SyntheticBaseballData, write_to_parquet


logger = logging.getLogger(__name__)

def generate_virtual_camera(config: dict) -> VirtualCamera:
    camera = VirtualCamera.from_mm_focal_length(
        focal_length_mm=2.74,      # Calculated from 70° FOV and 3.84mm sensor width
        sensor_width_mm=3.84,      # OV9281: 1280 pixels × 3.0μm
        rvec=np.array([np.pi/2, 0.0, 0.0]),  # Rotate +90° around X-axis to look toward mound
        position=np.array([0.0, -10, 3.5]),     # 10 ft behind plate (negative Y), 3.5 ft high (catcher POV)
        image_width=1280,          # Native resolution
        image_height=800
    )
    return camera

def generate_synthetic_data(num_samples: int, frames_dir: str) -> Iterator[SyntheticBaseballData]:
    """Generate synthetic data, skipping pitches with missing critical fields."""
    generated = 0
    attempts = 0
    max_attempts = num_samples * 3  # Try up to 3x to account for bad data
    
    while generated < num_samples and attempts < max_attempts:
        attempts += 1
        
        try:
            date, statcast_data = utils.pull_single_random_pitch_data()
            pitch_sample = statcast_data.iloc[0]
            pitch_dict = pitch_sample.to_dict()
            
            # Check for required fields
            required_fields = ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_pos_x', 'release_pos_z']
            missing = [f for f in required_fields if pitch_dict.get(f) is None]
            
            if missing:
                logger.warning(f"Skipping pitch from {date} - missing fields: {missing}")
                continue
            
            logger.info(f"Processing pitch {generated+1}/{num_samples} from date: {date}")
            
            t, x, y, z, r0, pred_pt, traj = generate_trajectory_points(pitch_dict)
            v_camera = generate_virtual_camera(None)
            end_time = traj.plate_intercept_time()
            projected_points = calculate_2d_points(traj, v_camera, num_frames=60, end_time=end_time)
            depths = v_camera.calculate_depth(traj(np.linspace(0.0, end_time, 60)))
            
            # Get ball radii and bounds for each frame
            radii, in_bounds = calculate_ball_data(projected_points, v_camera, depths, ball_radius_ft=0.12)
            frames = generate_video_frames(projected_points, v_camera, depths, ball_radius_ft=0.12, show_trail=False)
            
            # Reshape positions for easier use
            positions = projected_points.reshape(-1, 2)
            
            # Process frames for this pitch
            frames_yielded = 0
            for frame_idx, (position, radius, frame, is_in_bounds) in enumerate(zip(positions, radii, frames, in_bounds)):
                if not is_in_bounds:
                    continue  # Skip frames where ball is out of bounds
                
                # Save frame image with pitch grouping: pitch_{i}_frame_{j}.png
                frame_filename = f"pitch_{generated}_frame_{frame_idx}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_img.save(frame_path)
                
                synthetic_data = SyntheticBaseballData(
                    image_path=frame_path,
                    bbox_x=position[0],
                    bbox_y= position[1],
                    bbox_width=2 * radius,
                    bbox_height=2 * radius,
                    bbox_area=4 * radius * radius,
                    date=str(date)
                )
                yield synthetic_data
                frames_yielded += 1
            
            generated += 1
            logger.info(f"✓ Successfully generated pitch {generated}/{num_samples} with {frames_yielded} frames")
            
        except Exception as e:
            logger.error(f"Failed to process pitch (attempt {attempts}): {e}", exc_info=True)
            continue
    
    if generated < num_samples:
        logger.warning(f"Only generated {generated}/{num_samples} pitches after {attempts} attempts")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    TEST_BATCH_SIZE = 10
    logger.info(f"Generating {TEST_BATCH_SIZE} synthetic pitches...")
    
    # Create batch directory structure: batch_{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"data/synthetic_data/batch_{timestamp}"
    frames_dir = os.path.join(batch_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    logger.info(f"Output directory: {batch_dir}")
    
    # Collect all data entries
    all_data = []
    for entry in generate_synthetic_data(TEST_BATCH_SIZE, frames_dir):
        all_data.append(entry)
    
    logger.info(f"Collected {len(all_data)} frame annotations")
    
    # Save to parquet in batch directory
    output_parquet = os.path.join(batch_dir, "annotations.parquet")
    write_to_parquet(all_data, output_parquet, append=False)
    logger.info(f"✓ Saved {len(all_data)} annotations to {output_parquet}")
    logger.info(f"✓ Batch complete: {batch_dir}")




