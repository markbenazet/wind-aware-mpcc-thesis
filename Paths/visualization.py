import matplotlib.pyplot as plt
from Paths.curve import Path

def plot_waypoints_and_path(get_spline_curve):
    """
    Plots the generated path and waypoints.
    Args:
        curve (numpy.array): Array of points on the generated spline curve.
    """

    # Extract x and y coordinates from waypoints and curve
    waypoint_e = get_spline_curve[:, 1]
    waypoint_n = get_spline_curve[:, 0]

    # Plot waypoints and path
    plt.figure(figsize=(8, 6))
    plt.plot(waypoint_n, waypoint_e, 'r.-', label='Generated Path')
    plt.title('Generated Path and Waypoints')
    plt.xlabel('E')
    plt.ylabel('N')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot waypoints and path
#path_instance = Path()
#plot_waypoints_and_path(path_instance.get_bspline_curve())
#path_instance.precompute_and_save_data()
