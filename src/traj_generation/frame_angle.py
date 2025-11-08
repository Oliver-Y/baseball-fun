from src.traj_generation.trajectory_viz import generate_trajectory_points, Trajectory9P
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

def calculate_frames(traj: Trajectory9P, camera: VirtualCamera, num_frames: int = 30):
    # Realistically, you capture 3 or 4 frames?
    t = np.linspace(0.0, 0.6, num_frames) 
    point_3d = traj(t).astype(np.float64) #this is a (N,3) 
    logger.info(f"Shape: {point_3d.shape}")
    logger.info(f"Camera: {camera}")
    
    # Ensure rvec and tvec are float64 and have proper shape (3,1) for cv2.projectPoints
    rvec = camera.rvec.reshape(3, 1).astype(np.float64)
    tvec = camera.tvec.reshape(3, 1).astype(np.float64)
    
    projected_points, _ = cv2.projectPoints(point_3d, rvec, tvec, camera.intrinsic_matric, None)
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
                          trail_length: int = 5):
    """
    Generate video-like frames showing the ball moving through the camera view.
    
    Args:
        projected_points: (N, 1, 2) array of 2D projected points
        camera: VirtualCamera object
        trajectory_name: Name to display
        ball_radius: Radius of the ball in pixels
        trail_length: Number of previous positions to show as trail
        
    Returns:
        List of frame images
    """
    points_2d = projected_points.reshape(-1, 2)
    frames = []
    
    for i in range(len(points_2d)):
        # Create blank frame (dark background like night game)
        frame = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)  # Very dark gray background
        
        # Draw trail (previous positions)
        trail_start = max(0, i - trail_length)
        for j in range(trail_start, i):
            pt = tuple(points_2d[j].astype(int))
            if 0 <= pt[0] < camera.image_width and 0 <= pt[1] < camera.image_height:
                # Fade trail from old (dim) to recent (bright)
                alpha = (j - trail_start) / max(1, trail_length)
                color = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(frame, pt, max(2, ball_radius // 2), color, -1)
        
        # Draw current ball position (bright white)
        current_pt = tuple(points_2d[i].astype(int))
        if 0 <= current_pt[0] < camera.image_width and 0 <= current_pt[1] < camera.image_height:
            # Ball with slight glow effect
            cv2.circle(frame, current_pt, ball_radius + 2, (100, 100, 255), 2)  # Glow
            cv2.circle(frame, current_pt, ball_radius, (255, 255, 255), -1)  # Ball
        
        # Add frame info
        time_ms = int((i / len(points_2d)) * 600)  # Assuming ~600ms pitch
        cv2.putText(frame, f"{trajectory_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{len(points_2d)} ({time_ms}ms)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add strike zone reference (if ball is near plate)
        if i > len(points_2d) * 0.7:  # Last 30% of trajectory
            # Draw a simple strike zone box
            zone_center_x = camera.image_width // 2
            zone_center_y = camera.image_height // 2
            zone_width = camera.image_width // 6
            zone_height = camera.image_height // 4
            
            top_left = (zone_center_x - zone_width // 2, zone_center_y - zone_height // 2)
            bottom_right = (zone_center_x + zone_width // 2, zone_center_y + zone_height // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, "Strike Zone", (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        frames.append(frame)
    
    return frames


def save_as_gif(frames, output_path: str = "pitch_trajectory.gif", fps: int = 30):
    """
    Save frames as an animated GIF.
    
    Args:
        frames: List of frame images (numpy arrays)
        output_path: Path to save the GIF
        fps: Frames per second
    """
    from PIL import Image
    
    # Convert BGR (OpenCV) to RGB (PIL)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    # Save as GIF
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
    
    # Create a camera with Arducam B0332 specs (OV9281 sensor)
    # Position camera behind home plate, looking toward pitcher
    # rvec: rotation vector (radians), tvec: translation vector (camera position in feet)
    
    # Arducam B0332 specs:
    # - Resolution: 1280 × 800
    # - Sensor: OV9281 1/4" format (actual size: 3.84mm × 2.4mm)
    # - FOV: 70° (H)
    # - Calculated focal length: ~2.74mm → ~913 pixels
    
    # Camera positioned behind home plate, looking toward pitcher's mound
    # Rotation: rvec represents rotation around X, Y, Z axes (Rodrigues representation)
    # We need to rotate -90° around X-axis to point camera from looking up to looking forward
    camera = VirtualCamera.from_mm_focal_length(
        focal_length_mm=2.74,      # Calculated from 70° FOV and 3.84mm sensor width
        sensor_width_mm=3.84,      # OV9281: 1280 pixels × 3.0μm
        rvec=np.array([-np.pi/2, 0.0, 0.0]),  # Rotate -90° around X-axis to look toward mound
        tvec=np.array([0.0, -10.0, 3.5]),     # 10 ft behind plate, 3.5 ft high (catcher POV)
        image_width=1280,          # Native resolution
        image_height=800
    )
    
    logger.info(f"Camera: {camera}")
    projected_points = calculate_frames(traj, camera, num_frames=60)  # More frames for smoother animation
    logger.info(f"Projected points shape: {projected_points.shape}")
    logger.info(f"Sample points:\n{projected_points[:5]}")
    
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
    gif_path = "pitch_trajectory.gif"
    logger.info(f"Saving GIF with {len(frames)} frames...")
    save_as_gif(frames, gif_path, fps=30)
    logger.info(f"✓ Saved animated pitch to {gif_path}")
    
    # Save 3D trajectory visualization
    from src.traj_generation.trajectory_viz import viz_trajectory
    trajectory_3d_path = "pitch_trajectory_3d.png"
    logger.info(f"Saving 3D trajectory view...")
    viz_trajectory(pitch_data=pitch_dict, save_path=trajectory_3d_path)
    logger.info(f"✓ Saved 3D trajectory to {trajectory_3d_path}")
    
    # Also show the static camera view
    static_img = visualize_projected_trajectory(projected_points, camera, trajectory_name=pitch_type)
    cv2.imshow("Static Camera View (2D projection)", static_img)
    
    # Play the animation in a window
    logger.info("Playing animation... (press 'q' to quit)")
    for i, frame in enumerate(frames):
        cv2.imshow("Pitch Video Simulation", frame)
        key = cv2.waitKey(33)  # ~30 fps
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    logger.info("\n" + "="*60)
    logger.info("Output files generated:")
    logger.info(f"  - {gif_path} (animated camera view)")
    logger.info(f"  - {trajectory_3d_path} (3D trajectory plot)")
    logger.info("="*60)
    