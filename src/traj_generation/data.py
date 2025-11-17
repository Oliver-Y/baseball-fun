# Have this module pull down data in batches, generate the trajectories, move it to an image, 
# annotate the image, and store in different data formats
#Data formats: 
# 1. COCO format
# 2. YOLO format 
# What is the goal: Real-time inference, on-device for model within 0.5s latency
from typing import 
import numpy as np
from src import utils
import logging
from src.traj_generation.trajectory_viz import generate_trajectory_points

logger = logging.getLogger(__name__)


def generate_synthetic_data(num_samples: int) -> np.ndarray:
    for i in range(num_samples):
        date, statcast_data = utils.pull_single_random_pitch_data()
        logger.info(f"Pulled down data for date: {date}, Statcast Data: {statcast_data}")
        pitch_sample = statcast_data.iloc[0]
        pitch_dict = pitch_sample.to_dict()
        t, x, y, z, r0, pred_pt, traj = generate_trajectory_points(pitch_dict)
        #Create a synthetic baseball data object

    return data
    #We want to be able to batch sample statcast data
    pass

#Store generated synthetic data on disk for re-use
def store_data(data: np.ndarray, format: str, path: str):
    pass

if __name__ == "__main__":
    TEST_BATCH_SIZE = 100
    data = generate_synthetic_data(TEST_BATCH_SIZE)




