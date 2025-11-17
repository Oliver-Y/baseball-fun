from src.traj_generation.trajectory_viz import generate_trajectory_points, Trajectory9P, viz_trajectory
from src import utils
import numpy as np
import cv2
import logging
import argparse
from datetime import datetime
from src.virtual_camera import VirtualCamera
from rich.console import Console
from rich.table import Table
from PIL import Image

logger = logging.getLogger(__name__)

# Project 3D trajectory into 2D image plane
def calculate_2d_points(traj: Trajectory9P, camera: VirtualCamera, num_frames: int = 30, end_time: float = 0.6):
    t = np.linspace(0.0, end_time, num_frames) 
    point_3d = traj(t).astype(np.float64) #this is a (N,3) 
    projected_points, _ = cv2.projectPoints(point_3d, camera.extrinsics[:,:3], camera.extrinsics[:,3:4], camera.intrinsic_matric, None)
    return projected_points

def visualize_projected_trajectory(projected_points, camera: VirtualCamera, depths: np.ndarray, 
                                   trajectory_name: str = "Pitch", ball_radius_ft: float = 0.12):
    """
    Draw projected trajectory points on a blank image (static view) with perspective-correct sizing.
    
    Args:
        projected_points: 2D projected points (N, 2) or (N, 1, 2)
        camera: VirtualCamera object
        depths: Depth values for each point (N,)
        trajectory_name: Name to display
        ball_radius_ft: Physical ball radius in feet
    """
    # Create a blank image
    img = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
    
    points_2d = projected_points.reshape(-1, 2)
    
    # Draw trajectory path
    for pt1, pt2 in zip(points_2d[:-1], points_2d[1:]):
        pt1_int, pt2_int = tuple(pt1.astype(int)), tuple(pt2.astype(int))
        # Check if both points are within image bounds
        if (0 <= pt1_int[0] < camera.image_width and 0 <= pt1_int[1] < camera.image_height and
            0 <= pt2_int[0] < camera.image_width and 0 <= pt2_int[1] < camera.image_height):
            cv2.line(img, pt1_int, pt2_int, (0, 255, 0), 2)
    
    # Draw points
    n_points = len(points_2d)
    labels = {0: "Start", n_points - 1: "End"}
    
    for i, pt in enumerate(points_2d):
        pt_int = tuple(pt.astype(int))
        if not (0 <= pt_int[0] < camera.image_width and 0 <= pt_int[1] < camera.image_height):
            continue
            
        # Calculate perspective-correct radius
        point_radius = int((ball_radius_ft * camera.focal_length) / depths[i])
        point_radius = max(3, point_radius)  # Minimum 3 pixels for visibility
        
        # Color gradient from red (start) to blue (end)
        ratio = i / n_points
        color = (int(255 * (1 - ratio)), 0, int(255 * ratio))
        cv2.circle(img, pt_int, point_radius, color, -1)
        
        # Label first and last points
        if i in labels:
            cv2.putText(img, labels[i], (pt_int[0] + 10, pt_int[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add info text
    cv2.putText(img, f"{trajectory_name} - {len(points_2d)} points", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def generate_video_frames(projected_points, camera: VirtualCamera, 
                          depths: np.ndarray,
                          trajectory_name: str = "Pitch",
                          ball_radius_ft: float = 0.12,  # Baseball radius in feet (~2.9" diameter)
                          trail_length: int = 5,
                          ):
    """
    Generate video frames with perspective-correct ball sizing.
    
    Args:
        projected_points: 2D projected points (N, 2)
        camera: VirtualCamera object
        depths: Depth values for each point (N,)
        trajectory_name: Name for display
        ball_radius_ft: Physical ball radius in feet
        trail_length: Number of trailing frames to show
    """
    points_2d = projected_points.reshape(-1, 2)
    total_frames = len(points_2d)
    frames = []
    
    for i in range(len(points_2d)):
        frame = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)
        
        # Calculate perspective-correct ball radius in pixels
        # projected_radius = (radius * focal_length) / depth
        current_radius_raw = (ball_radius_ft * camera.focal_length) / depths[i]
        current_radius = int(current_radius_raw)
        current_radius = max(2, current_radius)  # Minimum 2 pixels
        
        # Debug: Print radius for every 10th frame
        logger.info(f"Frame {i}: depth={depths[i]:.2f}ft, focal_length={camera.focal_length:.1f}px, "
                    f"ball_radius={ball_radius_ft}ft, calculated_radius={current_radius_raw:.2f}px -> {current_radius}px")
        
        trail_start = max(0, i - trail_length)
        for j in range(trail_start, i):
            pt = tuple(points_2d[j].astype(int))
            if 0 <= pt[0] < camera.image_width and 0 <= pt[1] < camera.image_height:
                alpha = (j - trail_start) / max(1, trail_length)
                color = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                trail_radius = int((ball_radius_ft * camera.focal_length) / depths[j])
                trail_radius = max(2, trail_radius // 2)
                cv2.circle(frame, pt, trail_radius, color, -1)
        
        current_pt = tuple(points_2d[i].astype(int))
        if 0 <= current_pt[0] < camera.image_width and 0 <= current_pt[1] < camera.image_height:
            cv2.circle(frame, current_pt, current_radius + 2, (100, 100, 255), 2)  # Glow
            cv2.circle(frame, current_pt, current_radius, (255, 255, 255), -1)  # Ball
        
        time_ms = int((i / total_frames) * 600)  # Assuming ~600ms pitch
        cv2.putText(frame, f"{trajectory_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames} ({time_ms}ms)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Depth: {depths[i]:.1f}ft", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        frames.append(frame)
    
    return frames


def save_as_gif(frames, output_path: str = "pitch_trajectory.gif", fps: int = 30):
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,  # Duration per frame in ms
        loop=0  # Infinite loop
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize baseball pitch trajectory from camera perspective")
    parser.add_argument("--preview", action="store_true", help="Preview the video simulation interactively")
    args = parser.parse_args()
    
    # Logging is configured in src/__init__.py
    
    test_date, stat = utils.pull_single_random_pitch_data()
    logger.info(f"Date: {test_date}, Stat: {stat}")
    
    # Take the first pitch sample
    pitch_sample = stat.iloc[0]
    pitch_dict = pitch_sample.to_dict()
    t, x, y, z, r0, pred_pt, traj = generate_trajectory_points(pitch_dict)

    # Arducam B0332 specs:
    # - Resolution: 1280 × 800
    # - Sensor: OV9281 1/4" format (actual size: 3.84mm × 2.4mm)
    # - FOV: 70° (H)
    # - Calculated focal length: ~2.74mm → ~913 pixels
    # Camera must be BEHIND the plate (negative Y) to see the ball approaching
    # Trajectory: Y goes from ~53.9 (pitcher) to 0 (plate)
    # Camera at Y=-10 means 10 feet behind plate, so ball at Y=0 is 10 feet in front of camera
    # Camera height at 3.5 ft gives catcher's perspective
    camera = VirtualCamera.from_mm_focal_length(
        focal_length_mm=2.74,      # Calculated from 70° FOV and 3.84mm sensor width
        sensor_width_mm=3.84,      # OV9281: 1280 pixels × 3.0μm
        rvec=np.array([np.pi/2, 0.0, 0.0]),  # Rotate +90° around X-axis to look toward mound
        position=np.array([0.0, 0.0, 3.5]),     # 10 ft behind plate (negative Y), 3.5 ft high (catcher POV)
        image_width=1280,          # Native resolution
        image_height=800
    )

    end_time = traj.plate_intercept_time()

    #2D and depth points
    points_2d = calculate_2d_points(traj, camera, num_frames=60, end_time=end_time).reshape(-1, 2)
    depths = camera.calculate_depth(traj(np.linspace(0.0, end_time, 60)))
    
    # Print mapping between 3D trajectory points and 2D projected points
    console = Console()
    table = Table(title="3D TRAJECTORY POINT → 2D PROJECTED POINT MAPPING", show_header=True, header_style="bold magenta")
    
    table.add_column("Frame", style="cyan", justify="right")
    table.add_column("Time(s)", style="cyan", justify="right")
    table.add_column("3D X (ft)", style="green", justify="right")
    table.add_column("3D Y (ft)", style="green", justify="right")
    table.add_column("3D Z (ft)", style="green", justify="right")
    table.add_column("2D X (px)", style="yellow", justify="right")
    table.add_column("2D Y (px)", style="yellow", justify="right")
    table.add_column("In Bounds", justify="center")
    
    for i in range(len(points_2d)):
        t_val = i / 60 * end_time
        x_3d, y_3d, z_3d = traj(t_val)
        x_2d, y_2d = points_2d[i]
        in_bounds = (0 <= x_2d < camera.image_width and 0 <= y_2d < camera.image_height)
        in_bounds_str = "[green]✓[/green]" if in_bounds else "[red]✗[/red]"
        
        table.add_row(
            str(i+1),
            f"{t_val:.3f}",
            f"{x_3d:.3f}",
            f"{y_3d:.3f}",
            f"{z_3d:.3f}",
            f"{x_2d:.2f}",
            f"{y_2d:.2f}",
            in_bounds_str
        )
    
    console.print(table)
    
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)
    logger.info(f"Points in bounds: {in_bounds.sum()}/{len(points_2d)}")
    
    pitch_type = pitch_sample.get('pitch_type', 'Unknown')
    
    # Generate video frames
    logger.info("Generating video frames...")
    frames = generate_video_frames(
        points_2d, 
        camera,
        depths,
        trajectory_name=f"{pitch_type} Pitch",
        ball_radius_ft=0.12,  # Baseball radius in feet
        trail_length=8
    )
    
    # Save as GIF
    base_path = "data/pitch_trajectory" 
    #Grab time to timestamp the file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"{base_path}_{timestamp}.gif"
    save_as_gif(frames, gif_path, fps=30)
    logger.info(f"✓ Saved animated pitch to {gif_path}")
    
    # Save 3D trajectory visualization
    trajectory_3d_path = f"{base_path}_{timestamp}_3d.png"
    viz_trajectory(pitch_data=pitch_dict, save_path=trajectory_3d_path)
    logger.info(f"✓ Saved 3D trajectory to {trajectory_3d_path}")
    
    # Also show the static camera view
    static_img = visualize_projected_trajectory(points_2d, camera, depths, trajectory_name=pitch_type)
    static_view_path = f"{base_path}_{timestamp}_static_2d.png"
    cv2.imwrite(static_view_path, static_img)
    logger.info(f"✓ Saved static 2D camera view to {static_view_path}")
    
    # Preview the video simulation if requested
    if args.preview:
        cv2.imshow("Static Camera View (2D projection)", static_img)
        viz_trajectory(pitch_data=pitch_dict, save_path = None)
        
        # Play the animation in a window with arrow key navigation
        logger.info("Playing animation... (w/s: navigate frames, 'q': quit)")
        current_frame = 0
        while True:
            cv2.imshow("Pitch Video Simulation", frames[current_frame])
            key = cv2.waitKey(0) & 0xFF  # Get the key code (mask to get lower 8 bits)
            
            # Arrow keys: Up=82, Down=84 (on most systems)
            # Also support 'w'/'s' as alternatives
            if key == ord('w'): 
                current_frame = min(current_frame + 1, len(frames) - 1)
            elif key == ord('s'): 
                current_frame = max(current_frame - 1, 0)
            elif key == ord('q'): 
                break
        
        cv2.destroyAllWindows()
    else:
        logger.info("Skipping preview (use --preview flag to see interactive visualization)")
    
    logger.info("\n" + "="*60)
    logger.info("Output files generated:")
    logger.info(f"  - {gif_path} (animated camera view)")
    logger.info(f"  - {trajectory_3d_path} (3D trajectory plot)")
    logger.info(f"  - {static_view_path} (static 2D camera view)")
    logger.info("="*60)
    