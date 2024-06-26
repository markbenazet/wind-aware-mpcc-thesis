# path_points.py

import numpy as np

def generate_path_points(waypoints):
    path_points = []

    for i in range(len(waypoints) - 1):
        n_start, e_start = waypoints[i]
        n_end, e_end = waypoints[i + 1]

        # Number of points to add between each pair of waypoints (adjust as needed)
        num_points_between = 10

        # Generate points along the straight line segment
        segment_e = np.linspace(e_start, e_end, num_points_between + 2)[1:-1]  # Exclude start and end points
        segment_n = np.linspace(n_start, n_end, num_points_between + 2)[1:-1]  # Exclude start and end points

        path_points.extend(zip(segment_n, segment_e))

    return path_points
