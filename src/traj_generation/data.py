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
from src.traj_generation.trajectory_viz import generate_trajectory_points
from src.traj_generation.frame_angle import calculate_ball_data, generate_video_frames, calculate_2d_points
from src.virtual_camera import VirtualCamera
from src.traj_generation.model.db import SyntheticBaseballData


logger = logging.getLogger(__name__)

def generate_virtual_camera(config: dict) -> VirtualCamera:
    camera = VirtualCamera.from_mm_focal_length(
        focal_length_mm=2.74,      # Calculated from 70° FOV and 3.84mm sensor width
        sensor_width_mm=3.84,      # OV9281: 1280 pixels × 3.0μm
        rvec=np.array([np.pi/2, 0.0, 0.0]),  # Rotate +90° around X-axis to look toward mound
        position=np.array([0.0, 0.0, 3.5]),     # 10 ft behind plate (negative Y), 3.5 ft high (catcher POV)
        image_width=1280,          # Native resolution
        image_height=800
    )
    return camera

def generate_synthetic_data(num_samples: int) -> Iterator[SyntheticBaseballData]:
    for i in range(num_samples):
        date, statcast_data = utils.pull_single_random_pitch_data()
        logger.info(f"Pulled down data for date: {date}, Statcast Data: {statcast_data}")
        pitch_sample = statcast_data.iloc[0]
        pitch_dict = pitch_sample.to_dict()
        t, x, y, z, r0, pred_pt, traj = generate_trajectory_points(pitch_dict)
        #Generate virtual Camrea, pull in config extenrally usually but for now, just hardcode
        v_camera = generate_virtual_camera(None)
        #Create Traj
        end_time = traj.plate_intercept_time()
        projected_points = calculate_2d_points(traj, v_camera, num_frames=60, end_time=end_time)
        depths = v_camera.calculate_depth(traj(np.linspace(0.0, end_time, 60)))
        
        # Get ball radii and bounds for each frame
        radii, in_bounds = calculate_ball_data(projected_points, v_camera, depths, ball_radius_ft=0.12)
        frames = generate_video_frames(projected_points, v_camera, depths, ball_radius_ft=0.12, show_trail=False)
        
        # Reshape positions for easier use
        positions = projected_points.reshape(-1, 2)
        
        # Create output directory for this pitch
        output_dir = f"data/synthetic_data/{date}"
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx, (position, radius, frame, is_in_bounds) in enumerate(zip(positions, radii, frames, in_bounds)):
            if not is_in_bounds:
                continue  # Skip frames where ball is out of bounds
            
            # Save frame image
            frame_path = f"{output_dir}/frame_{frame_idx}.png"
            frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_img.save(frame_path)
            
            # Calculate bounding box (center - radius for top-left corner)
            center_x, center_y = position
            bbox_x = int(center_x - radius)
            bbox_y = int(center_y - radius)
            bbox_width = int(2 * radius)
            bbox_height = int(2 * radius)
            bbox_area = int(4 * radius * radius)
            
            # Create synthetic data entry
            synthetic_data = SyntheticBaseballData(
                image_path=frame_path,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                bbox_area=bbox_area,
                date=str(date)
            )
            logger.info(f"Generated {len(synthetic_entries)} frames for pitch from {date}")
            yield synthetic_data

#Store generated synthetic data to parquet
def save_to_parquet(data_entries: List[SyntheticBaseballData], output_path: str):
    """
    Save synthetic baseball data entries to a parquet file.
    
    Args:
        data_entries: List of SyntheticBaseballData objects
        output_path: Path to save the parquet file
    """
    # Convert to list of dicts
    data_dicts = [
        {
            'image_path': entry.image_path,
            'bbox_x': entry.bbox_x,
            'bbox_y': entry.bbox_y,
            'bbox_width': entry.bbox_width,
            'bbox_height': entry.bbox_height,
            'bbox_area': entry.bbox_area,
            'date': entry.date
        }
        for entry in data_entries
    ]
    
    # Create DataFrame and save to parquet
    df = pd.DataFrame(data_dicts)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(data_entries)} entries to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    TEST_BATCH_SIZE = 10
    logger.info(f"Generating {TEST_BATCH_SIZE} synthetic pitches...")
    
    # Collect all data entries
    all_data = list(generate_synthetic_data(TEST_BATCH_SIZE))
    
    # Save to parquet
    output_parquet = "data/synthetic_data/baseball_synthetic_data.parquet"
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    save_to_parquet(all_data, output_parquet)
    
    logger.info(f"✓ Generated {len(all_data)} total frames from {TEST_BATCH_SIZE} pitches")




