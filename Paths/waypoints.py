import math
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# List of waypoints defined as (north, east) tuples
path_points = []

# # Generate additional waypoints with curves
# num_waypoints = 80  # Number of additional waypoints for the circle
# radius = 50.0  # Radius of the circle
# circle_center_x = 0.0  # X-coordinate of the circle center
# circle_center_y = 0.0  # Y-coordinate of the circle center

# for i in range(num_waypoints):
#     angle = i * (2 * math.pi / num_waypoints)  # Angle between waypoints
#     x = circle_center_x + radius * math.cos(angle)  # X-coordinate of the waypoint
#     y = circle_center_y + radius * math.sin(angle)  # Y-coordinate of the waypoint
#     path_points.append((x, y))

# # Append the first waypoint of the circle to close it
# path_points.append(path_points[0])


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        new_point = (event.xdata, event.ydata)
        path_points.append(new_point)
        plt.plot(new_point[0], new_point[1], 'ro')  # Plot the selected point
        plt.draw()

def draw_spline():
    global path_points
    if len(path_points) > 2:
        x, y = zip(*path_points)
        tck, u = splprep([x, y], s=0)
        unew = np.linspace(0, 1.0, 100)
        out = splev(unew, tck)
        plt.plot(out[0], out[1], 'b-')  # Plot the spline
        plt.draw()
        
        # Update path_points with the spline points
        path_points = list(zip(out[0], out[1]))

def on_key(event):
    if event.key == ' ':
        draw_spline()

def select_points():
    fig, ax = plt.subplots()
    ax.set_title('Select waypoints by clicking on the map')
    ax.set_xlim(-400, 400)  # Adjust the limits as needed
    ax.set_ylim(-400, 400)  # Adjust the limits as needed
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Call the function to select points
select_points()

# Print the selected path points
print("Selected path points:")
for point in path_points:
    print(point)



