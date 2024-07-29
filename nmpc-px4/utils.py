import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as AnimationFunc
from matplotlib.collections import LineCollection

def plot_uav_trajectory_and_state(state_history, reference_history, solver_history, input_history, vector_p):
    """
    Plots the UAV trajectory, path points, reference points, and state variables and control inputs.
    Args:
        state_history (list): List of state vectors [I_n, I_e, B_v_x, B_v_y, I_yaw].
        path_points (list): List of path points [I_n, I_e].
        reference_history (list): List of reference points [I_n, I_e].
        input_history (list): List of control input vectors [B_a_x, B_a_y, I_yaw_rate].
        vector_p (numpy.array): Vector to be plotted as an arrow [I_n, I_e].
    """
    fig = plt.figure(figsize=(20, 15))

    # Left subplot for UAV Trajectory
    ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=4)
    ax1.plot([state[1] for state in state_history], [state[0] for state in state_history], 'b-', label='UAV Trajectory')
    ax1.plot([p[1] for p in reference_history], [p[0] for p in reference_history], 'r.', label='Reference Points')
    ax1.plot([s[1] for s in solver_history], [s[0] for s in solver_history], 'g.', label='Solver Points')
    ax1.arrow(0, 0, -3 * vector_p[1], -3 * vector_p[0], color='magenta', width=4.0, length_includes_head=True, head_width=4.0)
    ax1.set_xlabel('East')
    ax1.set_ylabel('North')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    # Right subplots for State Variables and Control Inputs
    axs = [plt.subplot2grid((3, 6), (i, 4), colspan=2) for i in range(3)]

    # Plot Velocity
    axs[0].plot([state[2] for state in state_history], 'b', label='V_x')
    axs[0].plot([state[3] for state in state_history], 'g', label='V_y')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('UAV Velocity')
    axs[0].legend()
    axs[0].grid()

    # Plot Yaw
    axs[1].plot([state[4] for state in state_history], 'g')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Yaw (radians)')
    axs[1].set_title('UAV Yaw')
    axs[1].grid()

    # Plot Control Inputs
    axs[2].plot([input[0] for input in input_history], 'b', label='Acceleration_x')
    axs[2].plot([input[1] for input in input_history], 'g', label='Acceleration_y')
    axs[2].plot([input[2] for input in input_history], 'r', label='Yaw_rate')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Input Value')
    axs[2].legend()
    axs[2].set_title('Control Inputs')
    axs[2].grid()

    plt.tight_layout()
    plt.show()

def plot_warm_start(optimal_history, reference_history, N_horizon, max_iterations):
    """
    Plots the UAV trajectory and reference points for warm start.
    Args:
        optimal_history (ndarray): Array of optimal state vectors with shape [N_horizon * iterations, 6].
        reference_history (list): List of reference points [I_n, I_e].
        N_horizon (int): Number of time steps in the horizon.
        max_iterations (int): Maximum number of iterations to plot.
    """
    fig = plt.figure(figsize=(20, 15))

    # Left subplot for UAV Trajectory
    ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=4)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i in range(min(max_iterations, len(optimal_history) // N_horizon)):
        start_idx = i * N_horizon
        end_idx = start_idx + N_horizon
        trajectory = optimal_history[start_idx:end_idx]
        
        ax1.plot([state[1] for state in trajectory], 
                 [state[0] for state in trajectory], 
                 color=colors[i % len(colors)], 
                 label=f'Iteration {i+1}')
        if i >= len(colors):
            colors.append(np.random.rand(3,))
    
    ax1.plot([p[1] for p in reference_history], [p[0] for p in reference_history], 'r.', label='Reference Points')
    ax1.set_xlabel('East')
    ax1.set_ylabel('North')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    plt.tight_layout()
    plt.show()

def plot_horizon_predictions(horizon_history, reference_history):
    plt.figure(figsize=(12, 8))
    plt.plot(reference_history[:, 0], reference_history[:, 1], 'k--', label='Reference Path')
    
    for i, horizon in enumerate(horizon_history):
        if i % 10 == 0:  # Plot every 10th horizon to avoid clutter
            plt.plot(horizon[:, 0], horizon[:, 1], 'r-', alpha=0.3)
    
    plt.plot([h[0, 0] for h in horizon_history], [h[0, 1] for h in horizon_history], 'b-', label='Vehicle Trajectory')
    
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Horizon Predictions Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_horizons(horizons, plane_states, N_horizon, max_simulation_time, horizon_time, sim_dt, path_points=None, interval=100, save_animation=False):
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Error checking and data preparation
    valid_horizons = []
    valid_states = []
    for i, (h, s) in enumerate(zip(horizons, plane_states)):
        if h.ndim == 2 and h.shape[1] >= 2 and isinstance(s, (list, np.ndarray)) and len(s) >= 2:
            valid_horizons.append(h)
            valid_states.append(s)
        else:
            print(f"Skipping invalid data at index {i}. Horizon shape: {h.shape if isinstance(h, np.ndarray) else type(h)}, State: {s}")
    
    if not valid_horizons or not valid_states:
        raise ValueError("No valid data to animate")
    
    # Plot path if provided
    if path_points is not None:
        ax.plot(path_points[:, 1], path_points[:, 0], 'k--', alpha=0.5, label='Reference Path')
    
    # Initialize a LineCollection for the horizons
    horizon_lines = LineCollection([], colors='blue', alpha=0.3)
    ax.add_collection(horizon_lines)
    
    # Initialize points and line for predicted and actual positions/trajectories
    current_predicted_point, = ax.plot([], [], 'bo', markersize=8, label='Predicted Position')
    actual_point, = ax.plot([], [], 'ro', markersize=10, label='Actual Position')
    actual_trajectory, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Actual Trajectory')
    
    # Set plot limits
    all_north = np.concatenate([h[:, 0] for h in valid_horizons] + [[s[0]] for s in valid_states])
    all_east = np.concatenate([h[:, 1] for h in valid_horizons] + [[s[1]] for s in valid_states])
    
    # Expand the range by 20% in all directions
    north_range = np.max(all_north) - np.min(all_north)
    east_range = np.max(all_east) - np.min(all_east)
    ax.set_xlim(np.min(all_east) - 0.2*east_range, np.max(all_east) + 0.2*east_range)
    ax.set_ylim(np.min(all_north) - 0.2*north_range, np.max(all_north) + 0.2*north_range)
    
    ax.set_xlabel('East position')
    ax.set_ylabel('North position')
    ax.set_title('Evolution of Predicted Horizons and Actual Trajectory')
    ax.legend()
    
    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        horizon_lines.set_segments([])
        current_predicted_point.set_data([], [])
        actual_point.set_data([], [])
        actual_trajectory.set_data([], [])
        time_text.set_text('')
        return horizon_lines, current_predicted_point, actual_point, actual_trajectory, time_text

    def update(frame):
        if frame < len(valid_horizons):
            current_time = frame * sim_dt
            horizon = valid_horizons[frame]
            actual_state = valid_states[frame]
            
            # Create line segments for the horizon
            segments = np.array([horizon[i:i+2, :2] for i in range(min(N_horizon-1, len(horizon)-1))])
            # Swap north and east for plotting
            segments = segments[:, :, [1, 0]]
            horizon_lines.set_segments(segments)
            
            # Update current predicted position
            current_predicted_point.set_data(horizon[0, 1], horizon[0, 0])  # East, North
            
            # Update actual position and trajectory
            actual_point.set_data(actual_state[1], actual_state[0])  # East, North
            actual_trajectory.set_data([state[1] for state in valid_states[:frame+1]],
                                       [state[0] for state in valid_states[:frame+1]])
            
            # Update time text
            time_text.set_text(f'Time: {current_time:.2f}s / {max_simulation_time:.2f}s\n'
                               f'Horizon: {horizon_time:.2f}s')
        
        return horizon_lines, current_predicted_point, actual_point, actual_trajectory, time_text

    total_frames = len(valid_horizons)
    anim = AnimationFunc(fig, update, frames=total_frames, init_func=init, blit=True, interval=interval)
    
    if save_animation:
        anim.save('horizon_evolution.gif', writer='pillow')
    
    return anim