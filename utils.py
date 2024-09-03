import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as AnimationFunc
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Cursor
from matplotlib.quiver import Quiver
import matplotlib.colors as colors
import matplotlib.patches as patches

def load_airplane_coords(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        coords = [list(map(float, row)) for row in reader]
    return np.array(coords)

def scale_coords(coords, scale_factor):
    # Center the coordinates
    centered_x = coords[:, 0] - 5.9389
    centered_y = coords[:, 1] - 5.9961
    
    # Scale the coordinates
    scaled_x = centered_x * scale_factor
    scaled_y = centered_y * scale_factor
    
    # Combine x and y coordinates
    scaled = np.column_stack((scaled_x, scaled_y))
    
    return scaled

def create_airplane_polygon(csv_file, scale_factor=0.01):
    coords = load_airplane_coords(csv_file)
    scaled_coords = scale_coords(coords, scale_factor)
    return scaled_coords

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def accumulate_angle(angles):
    """Convert regulated angles to continuous angles"""
    diff = np.diff(angles)
    diff[diff > np.pi] -= 2 * np.pi
    diff[diff < -np.pi] += 2 * np.pi
    return np.concatenate(([angles[0]], angles[0] + np.cumsum(diff)))


def plot_uav_trajectory_and_state(state_history, reference_history, solver_history, input_history, vector_p, cost_history):
    fig = plt.figure(figsize=(20, 30))

    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=5, colspan=4)
    ax1.plot([state[0] for state in state_history], [state[1] for state in state_history], 'b-', label='UAV Trajectory')
    ax1.plot([p[0] for p in reference_history], [p[1] for p in reference_history], 'k--', alpha=0.5, label='Reference Points')
    ax1.arrow(0, 0, 3 * vector_p[0,0], 3 * vector_p[1,0], color='magenta', width=4.0, length_includes_head=True, head_width=4.0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    axs = [plt.subplot2grid((6, 6), (i, 4), colspan=2) for i in range(6)]

    # Downsample state history to match input history length
    state_history_downsampled = state_history[::10]

    # Velocity plot
    axs[0].plot([state[2] for state in state_history_downsampled], 'b', label='V_x')
    axs[0].plot([state[3] for state in state_history_downsampled], 'g', label='V_y')
    axs[0].set_xlabel('Time Step (Downsampled)')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('UAV Velocity')
    axs[0].legend()
    axs[0].grid()

    # Yaw plot
    axs[1].plot([state[4] for state in state_history_downsampled], 'g')
    axs[1].set_xlabel('Time Step (Downsampled)')
    axs[1].set_ylabel('Yaw (radians)')
    axs[1].set_title('UAV Yaw')
    axs[1].grid()

    # Control inputs plot
    axs[2].plot([input[0] for input in input_history], 'b', label='Acceleration_x')
    axs[2].plot([input[1] for input in input_history], 'g', label='Acceleration_y')
    axs[2].plot([input[2] for input in input_history], 'r', label='Yaw_rate')
    axs[2].plot([input[3] for input in input_history], 'c', label='Speed')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Input Value')
    axs[2].legend()
    axs[2].set_title('Control Inputs')
    axs[2].grid()

    # Theta plot
    axs[3].plot([state[5] for state in state_history_downsampled], 'purple')
    axs[3].set_xlabel('Time Step (Downsampled)')
    axs[3].set_ylabel('Theta')
    axs[3].set_title('UAV Theta (Progress Along Path)')
    axs[3].grid()

    # Cost plot
    axs[4].plot(cost_history, 'orange')
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Cost')
    axs[4].set_title('MPC Cost')
    axs[4].grid()

    # Acceleration tracking plot
    # Calculate actual accelerations from downsampled state history
    
    
    axs[5].plot([input[0] for input in input_history], 'b--', label='ax_ref')
    axs[5].plot([state[8] for state in state_history], 'b-', label='ax_actual')
    axs[5].plot([input[1] for input in input_history], 'g--', label='ay_ref')
    axs[5].plot([state[6] for state in state_history], 'g-', label='ay_actual')
    axs[5].set_xlabel('Time Step')
    axs[5].set_ylabel('Acceleration (m/s^2)')
    axs[5].set_title('Acceleration Tracking')
    axs[5].legend()
    axs[5].grid()

    plt.tight_layout()

    # Add vertical lines to all subplots
    vlines = [ax.axvline(x=0, color='r', linestyle='--', visible=False) for ax in axs]
    
    # Add a cursor to the trajectory plot
    cursor = Cursor(ax1, useblit=True, color='red', linewidth=1)

    # Event handler for mouse clicks
    def on_click(event):
        if event.inaxes == ax1:
            x, y = event.xdata, event.ydata
            distances = [(x - state[0])**2 + (y - state[1])**2 for state in state_history]
            closest_index = distances.index(min(distances))
            
            for vline in vlines:
                vline.set_xdata(closest_index // 10)  # Adjust for downsampling
                vline.set_visible(True)
            
            fig.canvas.draw_idle()

    # Connect the event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

def plot_warm_start(optimal_history_list, reference_history, N_horizon, max_iterations):
    fig = plt.figure(figsize=(20, 15))

    ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=4)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, trajectory in enumerate(optimal_history_list[:max_iterations]):
        ax1.plot([state[0] for state in trajectory], 
                 [state[1] for state in trajectory], 
                 color=colors[i % len(colors)], 
                 label=f'Iteration {i+1}')
        if i >= len(colors):
            colors.append(np.random.rand(3,))
    
    ax1.plot([p[0] for p in reference_history], [p[1] for p in reference_history], 'r.', label='Reference Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    plt.tight_layout()
    plt.show()

def animate_horizons(horizons, plane_states, input_history, cost_history, N_horizon, max_simulation_time, horizon_time, sim_dt, params, path_points=None, interval=100, save_animation=False):
    fig = plt.figure(figsize=(20, 25))
    
    main_ax = plt.subplot2grid((5, 6), (0, 0), rowspan=5, colspan=4)
    axs = [plt.subplot2grid((5, 6), (i, 4), colspan=2) for i in range(5)]
    
    if path_points is not None:
        main_ax.plot(path_points[:, 0], path_points[:, 1], 'k--', alpha=0.5)
    
    horizon_lines = LineCollection([], colors='green', alpha=1.0, zorder=5)
    main_ax.add_collection(horizon_lines)
    
    actual_trajectory, = main_ax.plot([], [], 'r-', linewidth=2, alpha=0.7, zorder=10)
    
    # Create airplane polygon from CSV file
    airplane_coords = create_airplane_polygon('airplane.csv', scale_factor=1.0)
    airplane = Polygon(airplane_coords, closed=True, fc='black', ec='black', lw=1, zorder=1000)
    main_ax.add_patch(airplane)
    
    wx, wy = params.flatten()
    
    # Set up wind field
    x_range = main_ax.get_xlim()
    y_range = main_ax.get_ylim()
    n_arrows = 50  # Reduced number of arrows for clarity
    X = np.random.uniform(x_range[0], x_range[1], n_arrows)
    Y = np.random.uniform(y_range[0], y_range[1], n_arrows)
    
    # Adjust the scale of wind vectors
    wind_scale = 0.01 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])  # 1% of the plot size
    U = np.ones(n_arrows) * wx * wind_scale
    V = np.ones(n_arrows) * wy * wind_scale

    # Scale for arrow movement (can be different from visual length)
    move_scale = 0.1 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    U_move = np.ones(n_arrows) * wx * move_scale
    V_move = np.ones(n_arrows) * wy * move_scale

    lifetimes = np.random.uniform(0, 1, n_arrows)
    
    # Calculate wind speed for coloring
    speed = np.sqrt(U**2 + V**2)
    
    # Create a color map
    cmap = plt.get_cmap('coolwarm')
    norm = colors.Normalize(vmin=0, vmax=np.max(speed))

    # Create wind arrows
    wind_arrows = []
    for i in range(n_arrows):
        arrow = patches.FancyArrowPatch((X[i], Y[i]), (X[i] + U[i], Y[i] + V[i]),
                                        color='blue',
                                        arrowstyle='->',
                                        mutation_scale=10,
                                        linewidth=0.5,
                                        alpha=0.2,
                                        zorder=15)
        wind_arrows.append(arrow)
        main_ax.add_patch(arrow)

    # Calculate scalar wind magnitude
    wind_magnitude = np.linalg.norm(params)
    
    # Add wind vector legend
    wind_text = main_ax.text(0.02, 0.98, f'Wind: {wind_magnitude:.2f} m/s', 
                             transform=main_ax.transAxes, verticalalignment='top')
    
    # Apply initial rotation and translation
    initial_state = plane_states[0]
    initial_x, initial_y, initial_yaw = initial_state[0], initial_state[1], initial_state[4]
    initial_transform = Affine2D().rotate(-initial_yaw + np.pi).translate(initial_x, initial_y)
    airplane.set_transform(initial_transform + main_ax.transData)
    
    # Convert regulated yaw angles to continuous angles
    yaw_angles = [state[4] for state in plane_states]
    continuous_yaw = accumulate_angle(yaw_angles)
    
    all_x = np.concatenate([h[:, 0] for h in horizons] + [[s[0] for s in plane_states]])
    all_y = np.concatenate([h[:, 1] for h in horizons] + [[s[1] for s in plane_states]])
    
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    main_ax.set_xlim(np.min(all_x) - 0.2*x_range, np.max(all_x) + 0.2*x_range)
    main_ax.set_ylim(np.min(all_y) - 0.2*y_range, np.max(all_y) + 0.2*y_range)
    
    main_ax.set_xlabel('X position')
    main_ax.set_ylabel('Y position')
    main_ax.set_title('Evolution of Predicted Horizons and Actual Trajectory')
    
    time_text = main_ax.text(0.02, 0.95, '', transform=main_ax.transAxes)
    
    # Initialize subplots
    velocity_lines = [axs[0].plot([], [], label=label)[0] for label in ['V_x', 'V_y']]
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('UAV Velocity')
    axs[0].legend()
    axs[0].grid()

    yaw_line, = axs[1].plot([], [], 'g')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Yaw (radians)')
    axs[1].set_title('UAV Yaw')
    axs[1].grid()

    input_lines = [axs[2].plot([], [], label=label)[0] for label in ['Acceleration_x', 'Acceleration_y', 'Yaw_rate', 'Speed']]
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Input Value')
    axs[2].set_title('Control Inputs')
    axs[2].legend()
    axs[2].grid()

    theta_line, = axs[3].plot([], [], 'purple')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Theta')
    axs[3].set_title('UAV Theta (Progress Along Path)')
    axs[3].grid()

    cost_line, = axs[4].plot([], [], 'orange')
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Cost')
    axs[4].set_title('MPC Cost')
    axs[4].grid()

    def init():
        horizon_lines.set_segments([])
        actual_trajectory.set_data([], [])
        time_text.set_text('')
        airplane.set_xy(airplane_coords)
        for arrow in wind_arrows:
            arrow.set_alpha(0)
        for line in velocity_lines + input_lines + [yaw_line, theta_line, cost_line]:
            line.set_data([], [])
        return [horizon_lines, airplane, actual_trajectory, time_text] + wind_arrows + [wind_text] + velocity_lines + [yaw_line] + input_lines + [theta_line, cost_line]

    def update(frame):
        nonlocal X, Y, lifetimes
        current_time = frame * sim_dt
        horizon = horizons[frame]
        actual_state = plane_states[frame]
        
        segments = np.array([horizon[i:i+2, :2] for i in range(min(N_horizon-1, len(horizon)-1))])
        horizon_lines.set_segments(segments)
        
        # Update airplane position and rotation
        x, y = actual_state[0], actual_state[1]
        yaw = continuous_yaw[frame]  # Use continuous yaw angle

        lifetimes += 0.01  # Slower lifecycle
        new_arrows = lifetimes > 1
        lifetimes[new_arrows] = 0  # Reset lifetime for new arrows
        
        # Update positions for new arrows and move existing arrows
        X += U_move * 0.01  # Move arrows in the wind direction
        Y += V_move * 0.01
        X[new_arrows] = np.random.uniform(main_ax.get_xlim()[0], main_ax.get_xlim()[1], np.sum(new_arrows))
        Y[new_arrows] = np.random.uniform(main_ax.get_ylim()[0], main_ax.get_ylim()[1], np.sum(new_arrows))
        
        # Wrap arrows around the plot
        X = np.mod(X - main_ax.get_xlim()[0], main_ax.get_xlim()[1] - main_ax.get_xlim()[0]) + main_ax.get_xlim()[0]
        Y = np.mod(Y - main_ax.get_ylim()[0], main_ax.get_ylim()[1] - main_ax.get_ylim()[0]) + main_ax.get_ylim()[0]
        
        # Calculate arrow visibility based on lifetimes
        visibility = np.where(lifetimes <= 0.95, lifetimes, 0)  # Disappear at 95% of lifecycle
        
        for i, arrow in enumerate(wind_arrows):
            start = (X[i], Y[i])
            end = (X[i] + U[i], Y[i] + V[i])
            arrow.set_positions(start, end)
            arrow.set_alpha(visibility[i])

        wind_text.set_text(f'Wind: {wind_magnitude:.2f} m/s')
        
        # Rotate and translate the airplane
        transform = Affine2D().rotate(-yaw + np.pi).translate(x, y)
        airplane.set_transform(transform + main_ax.transData)
        
        actual_trajectory.set_data([state[0] for state in plane_states[:frame+1]],
                                   [state[1] for state in plane_states[:frame+1]])
        
        time_text.set_text(f'Time: {current_time:.2f}s / {max_simulation_time:.2f}s\n'
                           f'Horizon: {horizon_time:.2f}s')
        
        # Update subplots
        for i, line in enumerate(velocity_lines):
            line.set_data(range(frame+1), [state[2+i] for state in plane_states[:frame+1]])
        
        yaw_line.set_data(range(frame+1), [state[4] for state in plane_states[:frame+1]])
        
        for i, line in enumerate(input_lines):
            line.set_data(range(frame+1), [input[i] for input in input_history[:frame+1]])
        
        theta_line.set_data(range(frame+1), [state[5] for state in plane_states[:frame+1]])
        
        cost_line.set_data(range(frame+1), cost_history[:frame+1])
        
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
        
        return [horizon_lines, airplane, actual_trajectory, time_text] + wind_arrows + [wind_text] + velocity_lines + [yaw_line] + input_lines + [theta_line, cost_line]

    total_frames = len(horizons)
    anim = AnimationFunc(fig, update, frames=total_frames, init_func=init, blit=False, interval=interval)
    
    if save_animation:
        anim.save('horizon_evolution.gif', writer='pillow')
    
    plt.tight_layout()
    return anim

def plot_acceleration_tracking(state_history, input_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    
    # Plot x-acceleration
    ax1.plot([input[0] for input in input_history], 'b--', label='ax_ref')
    ax1.plot([state[8] for state in state_history], 'b-', label='ax_actual')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration X (m/s^2)')
    ax1.set_title('X-Acceleration Tracking')
    ax1.legend()
    ax1.grid(True)

    # Plot y-acceleration
    ax2.plot([input[1] for input in input_history], 'g--', label='ay_ref')
    ax2.plot([state[6] for state in state_history], 'g-', label='ay_actual')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration Y (m/s^2)')
    ax2.set_title('Y-Acceleration Tracking')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return fig