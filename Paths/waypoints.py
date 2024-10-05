import math
import matplotlib.pyplot as plt

# List of waypoints defined as (north, east) tuples
path_points = []

def interpolate_points(p1, p2, num_points=50):
    """Interpolate points between two given points."""
    x_values = [p1[0] + i * (p2[0] - p1[0]) / (num_points + 1) for i in range(1, num_points + 1)]
    y_values = [p1[1] + i * (p2[1] - p1[1]) / (num_points + 1) for i in range(1, num_points + 1)]
    return list(zip(x_values, y_values))

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        new_point = (event.xdata, event.ydata)
        if path_points:
            last_point = path_points[-1]
            interpolated_points = interpolate_points(last_point, new_point)
            path_points.extend(interpolated_points)
            for point in interpolated_points:
                plt.plot(point[0], point[1], 'bo')  # Plot the interpolated points
        path_points.append(new_point)
        plt.plot(new_point[0], new_point[1], 'ro')  # Plot the selected point
        plt.draw()

def select_points():
    fig, ax = plt.subplots()
    ax.set_title('Select waypoints by clicking on the map')
    ax.set_xlim(-400, 400)  # Adjust the limits as needed
    ax.set_ylim(-400, 400)  # Adjust the limits as needed
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# Call the function to select points
select_points()

# Print the selected path points
print("Selected path points:")
for point in path_points:
    print(point)



