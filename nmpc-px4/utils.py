import numpy as np
import matplotlib.pyplot as plt

def plot_uav_trajectory_and_state(state_history, reference_history, input_history, vector_p):
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