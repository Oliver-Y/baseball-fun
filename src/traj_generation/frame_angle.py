from src.traj_generation.trajectory_viz import generate_trajectory_points, Trajectory9P, viz_trajectory
from src import utils
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

#We need this notion of a Virtual Camera, that has an intrinsic matrix and an extrinsic matrix
class VirtualCamera:
    def __init__(self, focal_length: float, rvec: np.ndarray, tvec: np.ndarray, 
                 image_width: int = 1920, image_height: int = 1080):
        """
        Initialize a virtual camera.
        
        Args:
            focal_length: Focal length in PIXELS (not mm). For conversion see from_mm_focal_length()
            rvec: Rotation vector (3,) in radians
            tvec: Translation vector (3,) - camera position in world coordinates (feet)
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.focal_length = focal_length
        self.rvec = rvec
        self.tvec = tvec
        self.image_width = image_width
        self.image_height = image_height
        self.intrinsic_matric = self._construct_intrinsic_matrix(focal_length, image_width, image_height)
    
    @staticmethod
    def from_mm_focal_length(focal_length_mm: float, sensor_width_mm: float, 
                            rvec: np.ndarray, tvec: np.ndarray,
                            image_width: int = 1920, image_height: int = 1080):
        """
        Create a VirtualCamera from focal length in millimeters.
        
        Args:
            focal_length_mm: Physical focal length in mm (e.g., 50mm lens)
            sensor_width_mm: Physical sensor width in mm (e.g., 36mm for full frame)
            rvec, tvec: Camera pose
            image_width, image_height: Image dimensions in pixels
            
        Example:
            # 50mm lens on full-frame sensor (36mm)
            camera = VirtualCamera.from_mm_focal_length(50, 36, rvec, tvec)
        """
        focal_length_pixels = focal_length_mm * (image_width / sensor_width_mm)
        return VirtualCamera(focal_length_pixels, rvec, tvec, image_width, image_height)
    
    #Assumes no skew, principal point at image center
    def _construct_intrinsic_matrix(self, focal_length: float, width: int, height: int):
        cx = width / 2.0   # principal point x (image center)
        cy = height / 2.0  # principal point y (image center)
        return np.array([[focal_length, 0, cx], 
                        [0, focal_length, cy], 
                        [0, 0, 1]], dtype=np.float64)
    
    def __str__(self): 
        return f"VirtualCamera(focal_length={self.focal_length}px, rvec={self.rvec}, tvec={self.tvec}, image_size=({self.image_width}x{self.image_height}))"

def build_extrinsics_from_pose(rvec_axis_angle, C_world):
    """
    Build rotation matrix R and translation vector t from camera pose.
    
    Args:
        rvec_axis_angle: Rotation vector (axis-angle, OpenCV convention), e.g. [-pi/2, 0, 0] for Rx(-90°)
        C_world: Camera position in world coordinates (3,)
    
    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector in camera coordinates (3, 1)
    """
    # rvec is axis-angle (OpenCV convention), e.g. [-pi/2, 0, 0] for Rx(-90°)
    R, _ = cv2.Rodrigues(rvec_axis_angle.reshape(3,1).astype(np.float64))
    t = -R @ C_world.reshape(3,1).astype(np.float64)
    return R, t

def calculate_frames(traj: Trajectory9P, camera: VirtualCamera, num_frames: int = 30):
    # Realistically, you capture 3 or 4 frames?
    t = np.linspace(0.0, 0.6, num_frames) 
    point_3d = traj(t).astype(np.float64) #this is a (N,3) 
    logger.info(f"Shape: {point_3d.shape}")
    logger.info(f"Camera: {camera}")
    
    # Build extrinsics from pose: camera.tvec is C_world (camera position in world coordinates)
    R, t_cam = build_extrinsics_from_pose(camera.rvec, camera.tvec)
    rvec = camera.rvec.reshape(3, 1).astype(np.float64)
    
    projected_points, _ = cv2.projectPoints(point_3d, rvec, t_cam, camera.intrinsic_matric, None)
    
    logger.info(f"Projected points shape: {projected_points.shape}")
    return projected_points

def visualize_projected_trajectory(projected_points, camera: VirtualCamera, trajectory_name: str = "Pitch"):
    """
    Draw projected trajectory points on a blank image (static view).
    """
    # Create a blank image
    img = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
    
    # Reshape projected points from (N, 1, 2) to (N, 2)
    points_2d = projected_points.reshape(-1, 2)
    
    # Draw trajectory path
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))
        # Check if points are within image bounds
        if (0 <= pt1[0] < camera.image_width and 0 <= pt1[1] < camera.image_height and
            0 <= pt2[0] < camera.image_width and 0 <= pt2[1] < camera.image_height):
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw points
    for i, pt in enumerate(points_2d):
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < camera.image_width and 0 <= pt_int[1] < camera.image_height:
            # Color gradient from red (start) to blue (end)
            color_ratio = i / len(points_2d)
            color = (int(255 * (1 - color_ratio)), 0, int(255 * color_ratio))
            cv2.circle(img, pt_int, 5, color, -1)
            
            # Label first, middle, and last points
            if i == 0:
                cv2.putText(img, "Start", (pt_int[0] + 10, pt_int[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif i == len(points_2d) - 1:
                cv2.putText(img, "End", (pt_int[0] + 10, pt_int[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add info text
    cv2.putText(img, f"{trajectory_name} - {len(points_2d)} points", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def generate_video_frames(projected_points, camera: VirtualCamera, 
                          trajectory_name: str = "Pitch",
                          ball_radius: int = 8,
                          trail_length: int = 5,
                          ):
    points_2d = projected_points.reshape(-1, 2)
    total_frames = len(points_2d)
    frames = []
    
    for i in range(len(points_2d)):
        frame = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)
        
        trail_start = max(0, i - trail_length)
        for j in range(trail_start, i):
            pt = tuple(points_2d[j].astype(int))
            if 0 <= pt[0] < camera.image_width and 0 <= pt[1] < camera.image_height:
                alpha = (j - trail_start) / max(1, trail_length)
                color = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(frame, pt, max(2, ball_radius // 2), color, -1)
        
        current_pt = tuple(points_2d[i].astype(int))
        if 0 <= current_pt[0] < camera.image_width and 0 <= current_pt[1] < camera.image_height:
            cv2.circle(frame, current_pt, ball_radius + 2, (100, 100, 255), 2)  # Glow
            cv2.circle(frame, current_pt, ball_radius, (255, 255, 255), -1)  # Ball
        
        time_ms = int((i / total_frames) * 600)  # Assuming ~600ms pitch
        cv2.putText(frame, f"{trajectory_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames} ({time_ms}ms)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        frames.append(frame)
    
    return frames


def save_as_gif(frames, output_path: str = "pitch_trajectory.gif", fps: int = 30):
    from PIL import Image
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,  # Duration per frame in ms
        loop=0  # Infinite loop
    )
    logger.info(f"Saved GIF to {output_path}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
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
        rvec=np.array([-np.pi/2, 0.0, 0.0]),  # Rotate -90° around X-axis to look toward mound
        tvec=np.array([0.0, -20.0, 3.5]),     # 10 ft behind plate, 3.5 ft high (catcher POV)
        image_width=1280,          # Native resolution
        image_height=800
    )
    
    logger.info(f"Camera: {camera}")
    
    # Debug: Verify rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(camera.rvec.reshape(3, 1))
    logger.info(f"Rotation matrix:\n{rotation_matrix}")
    logger.info(f"Camera position (tvec): {camera.tvec}")
    
    # Test projection of a known point: pitcher's mound center (should be roughly at image center if camera is aligned)
    test_point_mound = np.array([[0.0, 60.5, 6.0]], dtype=np.float64)  # Center of mound, 6 ft high
    R_test, t_test = build_extrinsics_from_pose(camera.rvec, camera.tvec)
    test_proj_mound, _ = cv2.projectPoints(
        test_point_mound, 
        camera.rvec.reshape(3, 1).astype(np.float64), 
        t_test, 
        camera.intrinsic_matric, 
        None
    )
    logger.info(f"Test: Mound center projects to: ({test_proj_mound[0, 0, 0]:.2f}, {test_proj_mound[0, 0, 1]:.2f})")
    
    # Debug: Print 3D trajectory points
    t_debug = np.linspace(0.0, 0.6, 60)
    point_3d_debug = traj(t_debug)
    logger.info(f"3D Trajectory Points (first 5 and last 5):")
    logger.info(f"First 5:\n{point_3d_debug[:5]}")
    logger.info(f"Last 5:\n{point_3d_debug[-5:]}")
    logger.info(f"Y range (toward plate): {point_3d_debug[:, 1].min():.2f} to {point_3d_debug[:, 1].max():.2f}")
    logger.info(f"Z range (height): {point_3d_debug[:, 2].min():.2f} to {point_3d_debug[:, 2].max():.2f}")
    logger.info(f"X range (horizontal): {point_3d_debug[:, 0].min():.2f} to {point_3d_debug[:, 0].max():.2f}")
    logger.info(f"Release point (r0): X={r0[0]:.3f}, Y={r0[1]:.3f}, Z={r0[2]:.3f}")
    
    projected_points = calculate_frames(traj, camera, num_frames=60)  # More frames for smoother animation
    logger.info(f"Projected points shape: {projected_points.shape}")
    
    # Check which points are in bounds
    points_2d = projected_points.reshape(-1, 2)
    
    # Print mapping between 3D trajectory points and 2D projected points
    logger.info("\n" + "="*80)
    logger.info("3D TRAJECTORY POINT -> 2D PROJECTED POINT MAPPING")
    logger.info("="*80)
    logger.info(f"{'Frame':<6} {'Time(s)':<8} {'3D X (ft)':<12} {'3D Y (ft)':<12} {'3D Z (ft)':<12} {'2D X (px)':<12} {'2D Y (px)':<12} {'In Bounds':<10}")
    logger.info("-"*80)
    
    for i in range(len(point_3d_debug)):
        t_val = t_debug[i]
        x_3d, y_3d, z_3d = point_3d_debug[i]
        x_2d, y_2d = points_2d[i]
        in_bounds_flag = "Yes" if (0 <= x_2d < camera.image_width and 0 <= y_2d < camera.image_height) else "No"
        logger.info(f"{i+1:<6} {t_val:<8.3f} {x_3d:<12.3f} {y_3d:<12.3f} {z_3d:<12.3f} {x_2d:<12.2f} {y_2d:<12.2f} {in_bounds_flag:<10}")
    
    logger.info("="*80)
    
    logger.info(f"First projected point (release): ({points_2d[0, 0]:.2f}, {points_2d[0, 1]:.2f})")
    logger.info(f"Last projected point (plate): ({points_2d[-1, 0]:.2f}, {points_2d[-1, 1]:.2f})")
    logger.info(f"Image center: ({camera.image_width/2:.2f}, {camera.image_height/2:.2f})")
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)
    logger.info(f"Points in bounds: {in_bounds.sum()}/{len(points_2d)}")
    logger.info(f"X range (projected): {points_2d[:, 0].min():.2f} to {points_2d[:, 0].max():.2f}")
    logger.info(f"Y range (projected): {points_2d[:, 1].min():.2f} to {points_2d[:, 1].max():.2f}")
    
    pitch_type = pitch_sample.get('pitch_type', 'Unknown')
    
    # Generate video frames
    logger.info("Generating video frames...")
    frames = generate_video_frames(
        projected_points, 
        camera, 
        trajectory_name=f"{pitch_type} Pitch",
        ball_radius=10,
        trail_length=8
    )
    
    # Save as GIF
    gif_path = "data/pitch_trajectory.gif"
    logger.info(f"Saving GIF with {len(frames)} frames...")
    save_as_gif(frames, gif_path, fps=30)
    logger.info(f"✓ Saved animated pitch to {gif_path}")
    
    # Save 3D trajectory visualization
    trajectory_3d_path = "data/pitch_trajectory_3d.png"
    logger.info(f"Saving 3D trajectory view...")
    viz_trajectory(pitch_data=pitch_dict, save_path=trajectory_3d_path)
    logger.info(f"✓ Saved 3D trajectory to {trajectory_3d_path}")
    
    # Also show the static camera view
    static_img = visualize_projected_trajectory(projected_points, camera, trajectory_name=pitch_type)
    static_view_path = "data/pitch_trajectory_static_2d.png"
    cv2.imwrite(static_view_path, static_img)
    logger.info(f"✓ Saved static 2D camera view to {static_view_path}")
    cv2.imshow("Static Camera View (2D projection)", static_img)
    
    # Play the animation in a window with arrow key navigation
    logger.info("Playing animation... (Up/Down arrows: navigate, 'q': quit)")
    current_frame = 0
    while True:
        cv2.imshow("Pitch Video Simulation", frames[current_frame])
        key = cv2.waitKey(0) & 0xFF  # Get the key code (mask to get lower 8 bits)
        
        # Arrow keys: Up=82, Down=84 (on most systems)
        # Also support 'w'/'s' as alternatives
        if key == 82 or key == ord('w') or key == ord('W'):  # Up arrow or 'w'
            current_frame = min(current_frame + 1, len(frames) - 1)
        elif key == 84 or key == ord('s') or key == ord('S'):  # Down arrow or 's'
            current_frame = max(current_frame - 1, 0)
        elif key == ord('q') or key == ord('Q'):  # Quit
            break
        elif key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    
    logger.info("\n" + "="*60)
    logger.info("Output files generated:")
    logger.info(f"  - {gif_path} (animated camera view)")
    logger.info(f"  - {trajectory_3d_path} (3D trajectory plot)")
    logger.info(f"  - {static_view_path} (static 2D camera view)")
    logger.info("="*60)
    