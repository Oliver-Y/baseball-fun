import numpy as np
import cv2

class VirtualCamera:
    def __init__(self, focal_length: float, rvec: np.ndarray, position: np.ndarray, 
                 image_width: int = 1920, image_height: int = 1080):
        """
        Initialize a virtual camera.
        
        Args:
            focal_length: Focal length in PIXELS (not mm). For conversion see from_mm_focal_length()
            rvec: axis angle rotation vector in radians
            position: Camera position in world coordinates (3,)
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.focal_length = focal_length
        self.rvec = rvec
        self.position = position
        self.image_width = image_width
        self.image_height = image_height
        self.intrinsic_matric = self._construct_intrinsic_matrix(focal_length, image_width, image_height)
        self.extrinsics = self._construct_extrinsics()
    
    @staticmethod
    def from_mm_focal_length(focal_length_mm: float, sensor_width_mm: float, 
                            rvec: np.ndarray, position: np.ndarray,
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
        return VirtualCamera(focal_length_pixels, rvec, position, image_width, image_height)

    def world_to_camera(self, points_3d: np.ndarray):
        """
        Transform 3D world coordinates to camera coordinate system.
        
        Args:
            points_3d: (N, 3) array or (3,) array of world coordinates
            
        Returns:
            (N, 3) array or (3,) array of camera coordinates
        """
        # Handle single point vs multiple points
        single_point = (points_3d.ndim == 1)
        if single_point:
            points_3d = points_3d.reshape(1, -1)
        
        # Build 4x4 transformation matrix
        extrinsic_4x4 = np.vstack([self.extrinsics, [0, 0, 0, 1]])
        
        # Add homogeneous coordinate: (N, 3) -> (N, 4)
        points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Transform: (4, 4) @ (N, 4).T = (4, N) -> transpose to (N, 4)
        points_camera = (extrinsic_4x4 @ points_homogeneous.T).T
        
        # Drop homogeneous coordinate and return
        result = points_camera[:, :3]
        return result[0] if single_point else result
    
    def calculate_depth(self, points_3d: np.ndarray):
        """
        Calculate depth (distance along optical axis) for 3D points.
        
        Args:
            points_3d: (N, 3) array or (3,) array of world coordinates
            
        Returns:
            (N,) array or float of depth values
        """
        camera_coords = self.world_to_camera(points_3d)
        # Handle both single point and multiple points
        # By convention, camera Z-axis is the viewing direction (depth)
        if camera_coords.ndim == 1:
            return camera_coords[2]
        return camera_coords[:, 2]
    
    #Assumes no skew, principal point at image center
    def _construct_intrinsic_matrix(self, focal_length: float, width: int, height: int):
        cx = width / 2.0   # principal point x (image center)
        cy = height / 2.0  # principal point y (image center)
        return np.array([[focal_length, 0, cx], 
                        [0, focal_length, cy], 
                        [0, 0, 1]], dtype=np.float64)
    def _construct_extrinsics(self):
        R, _ = cv2.Rodrigues(self.rvec.reshape(3,1).astype(np.float64))
        t = -R @ self.position.reshape(3,1).astype(np.float64)
        return np.concatenate([R, t], axis=1)
    
    def __str__(self): 
        return f"VirtualCamera(focal_length={self.focal_length}px, rvec={self.rvec}, tvec={self.tvec}, image_size=({self.image_width}x{self.image_height}))"