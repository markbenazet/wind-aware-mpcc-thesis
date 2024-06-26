# Paths/path_manager.py

from . import path_points  # Import path_points module from the current package
from . import waypoints   # Import waypoints module for path_points
from . import visualization   # Import visualization module for plotting

def generate_path_from_waypoints():
    # Generate path using straight line segments
    path_points_list = path_points.generate_path_points(waypoints.path_points)

    # Visualize waypoints and generated path (optional)
    # visualization.plot_waypoints_and_path(waypoints.path_points, path_points_list)

    return path_points_list
