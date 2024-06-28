from . import path_points  # Import path_points module from the current package
from . import waypoints   # Import waypoints module for path_points
from . import visualization   # Import visualization module for plotting
class PathManager:
    def __init__(self):
        self.path = None
        self.lookahead_distance = 40 

    def generate_path_from_waypoints(self):
        # Generate path using cubic interpolation
        self.path = path_points.generate_path_points(waypoints.path_points)

        # Visualize waypoints and generated path (optional)
        visualization.plot_waypoints_and_path(waypoints.path_points, self.path)

        return self.path

    def get_reference_point(self, current_position):
        if self.path is None:
            raise ValueError("Path has not been generated. Call generate_path_from_waypoints() first.")
        
        return path_points.get_lookahead_point(current_position, self.path, self.lookahead_distance)

# Create an instance of PathManager
path_manager = PathManager()
