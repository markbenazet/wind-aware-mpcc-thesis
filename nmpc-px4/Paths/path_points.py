from scipy.interpolate import interp1d
import numpy as np

def generate_path_points(waypoints, num_points=1000):
    waypoints = np.array(waypoints)

    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required to generate a path.")

    # Interpolation threshold
    t = np.linspace(0, 1, len(waypoints))

    # Choose interpolation method based on number of waypoints
    kind = 'linear' if len(waypoints) < 4 else 'cubic'

    # Create interpolation functions for north and east coordinates
    n_interp = interp1d(t, waypoints[:, 0], kind=kind)
    e_interp = interp1d(t, waypoints[:, 1], kind=kind)

    # Generate more points along the path
    t_fine = np.linspace(0, 1, num_points)
    n_path = n_interp(t_fine)
    e_path = e_interp(t_fine)

    return np.column_stack((n_path, e_path))

def get_lookahead_point(path_points, current_position, lookahead_distance,):
    current_position = np.array(current_position)
    path_points = np.array(path_points)
    
    if path_points.ndim == 1:
        # If path_points is a single point, return it
        print("Path points is a single point")
        return path_points

    # Find the closest point on the path
    distance = np.linalg.norm(path_points - current_position, axis=1)
    closest_point_idx = np.argmin(distance)

    # Find the lookahead point on the path
    lookahead_point_idx = closest_point_idx
    distance = 0
    while distance < lookahead_distance and lookahead_point_idx < len(path_points) - 1:
        lookahead_point_idx += 1
        distance += np.linalg.norm(path_points[lookahead_point_idx] - path_points[lookahead_point_idx - 1])

    return path_points[lookahead_point_idx]