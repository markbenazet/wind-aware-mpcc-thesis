# path_points.py

import numpy as np

def generate_path_points(waypoints):
    path_points = []

    for i in range(len(waypoints) - 1):
        x_start, y_start = waypoints[i]
        x_end, y_end = waypoints[i + 1]

        # Number of points to add between each pair of waypoints (adjust as needed)
        num_points_between = 10

        # Generate points along the straight line segment
        segment_x = np.linspace(x_start, x_end, num_points_between + 2)[1:-1]  # Exclude start and end points
        segment_y = np.linspace(y_start, y_end, num_points_between + 2)[1:-1]  # Exclude start and end points

        path_points.extend(zip(segment_x, segment_y))

    # Add the last waypoint to the path
    path_points.append(waypoints[-1])

    return path_points
