import matplotlib.pyplot as plt

def plot_waypoints_and_path(waypoints, path_points):
    # Extract x and y coordinates from waypoints and path_points
    waypoint_n = [point[0] for point in waypoints]
    waypoint_e = [point[1] for point in waypoints]

    # Extract east (x) and north (y) coordinates for plotting
    path_e = [point[1] for point in path_points]
    path_n = [point[0] for point in path_points]

    # Plot waypoints and path
    plt.figure(figsize=(8, 6))
    plt.plot(waypoint_e, waypoint_n, 'bo-', label='Waypoints')
    plt.plot(path_e, path_n, 'r.-', label='Generated Path')
    plt.title('Generated Path and Waypoints')
    plt.xlabel('E')
    plt.ylabel('N')
    plt.legend()
    plt.grid(True)
    plt.show()
