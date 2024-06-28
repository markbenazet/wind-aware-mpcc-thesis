import math

# List of waypoints defined as (north, east) tuples
path_points = [
    (0.0, 0.0),
    (400.0, 20.0),
    (800.0, 40.0),
]


# # Generate additional waypoints with curves
# num_waypoints = 5  # Number of additional waypoints
# radius = 100.0  # Radius of the curve

# for i in range(1, num_waypoints + 1):
#     angle = i * (2 * math.pi / num_waypoints)  # Angle between waypoints
#     x = 800.0 + radius * math.cos(angle)  # X-coordinate of the waypoint
#     y = 40.0 + radius * math.sin(angle)  # Y-coordinate of the waypoint
#     path_points.append((x, y))
