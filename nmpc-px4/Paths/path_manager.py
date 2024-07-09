from . import path_points  # Import path_points module from the current package
from . import waypoints   # Import waypoints module for path_points
from . import visualization   # Import visualization module for plotting
import numpy as np
class PathManager:
    def __init__(self):
        self.path = None

    def generate_path_from_waypoints(self):
        # Generate path using cubic interpolation
        self.path = path_points.generate_path_points(waypoints.path_points)

        # Visualize waypoints and generated path
        visualization.plot_waypoints_and_path(waypoints.path_points, self.path)

        return self.path

    def get_reference_point(self, current_position, lookahead_distance, previous_reference=None):
        if self.path is None:
            raise ValueError("Path has not been generated. Call generate_path_from_waypoints() first.")
        
        # Find the lookahead point if there's no previous reference
        if previous_reference is None:
            return path_points.get_lookahead_point(self.path, current_position, lookahead_distance)
        
        # Ensure we continue forward from the previous reference point
        previous_ref_idx = np.argmin(np.linalg.norm(self.path - previous_reference, axis=1))
        current_position_idx = np.argmin(np.linalg.norm(self.path - current_position, axis=1))

        if current_position_idx >= previous_ref_idx:
            return path_points.get_lookahead_point(self.path[current_position_idx:], current_position, lookahead_distance)
        else:
            # If current position index is behind the previous reference index, use the full path
            return path_points.get_lookahead_point(self.path, current_position, lookahead_distance)
    
    def get_path_tangent(self,reference_point):
        if self.path is None:
            raise ValueError("Path has not been generated. Call generate_path_from_waypoints() first.")
        
        return path_points.get_path_tangent(self.path, reference_point)
    

# Create an instance of PathManager
path_manager = PathManager()
