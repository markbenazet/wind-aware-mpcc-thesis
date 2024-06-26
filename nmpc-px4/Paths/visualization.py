import matplotlib.pyplot as plt

def plot_waypoints_and_path(waypoints, path_points):
    # Extract x and y coordinates from waypoints and path_points
    waypoint_x = [point[0] for point in waypoints]
    waypoint_y = [point[1] for point in waypoints]

    path_x = [point[0] for point in path_points]
    path_y = [point[1] for point in path_points]

    # Plot waypoints and path
    plt.figure(figsize=(8, 6))
    plt.plot(waypoint_x, waypoint_y, 'bo-', label='Waypoints')
    plt.plot(path_x, path_y, 'r.-', label='Generated Path')
    plt.title('Generated Path and Waypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
