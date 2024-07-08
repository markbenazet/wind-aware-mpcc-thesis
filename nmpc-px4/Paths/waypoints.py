import math

# List of waypoints defined as (north, east) tuples
path_points = [
    (0.0, 0.0),
    (400.0, 0.0),
    (800.0, 0.0),
    (1200.0, 0.0),
    (1600.0 , 0.0),
    (2000.0, 0.0),
    (2400.0, 0.0)
]


# # Generate additional waypoints with curves
# num_waypoints = 60  # Number of additional waypoints for the circle
# radius = 200.0  # Radius of the circle
# circle_center_x = 0.0  # X-coordinate of the circle center
# circle_center_y = 0.0  # Y-coordinate of the circle center

# for i in range(num_waypoints):
#     angle = i * (2 * math.pi / num_waypoints)  # Angle between waypoints
#     x = circle_center_x + radius * math.cos(angle)  # X-coordinate of the waypoint
#     y = circle_center_y + radius * math.sin(angle)  # Y-coordinate of the waypoint
#     path_points.append((x, y))

# # Append the first waypoint of the circle to close it
# path_points.append(path_points[0])
