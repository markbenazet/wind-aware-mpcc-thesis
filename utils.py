import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as AnimationFunc
from matplotlib.collections import LineCollection


import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

def plot_uav_trajectory_and_state(state_history, reference_history, solver_history, input_history, vector_p, cost_history):
    fig = plt.figure(figsize=(20, 25))

    ax1 = plt.subplot2grid((5, 6), (0, 0), rowspan=5, colspan=4)
    ax1.plot([state[0] for state in state_history], [state[1] for state in state_history], 'b-', label='UAV Trajectory')
    ax1.plot([p[0] for p in reference_history], [p[1] for p in reference_history], 'r.', label='Reference Points')
    ax1.arrow(0, 0, 3 * vector_p[0,0], 3 * vector_p[1,0], color='magenta', width=4.0, length_includes_head=True, head_width=4.0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    axs = [plt.subplot2grid((5, 6), (i, 4), colspan=2) for i in range(5)]

    axs[0].plot([state[2] for state in state_history], 'b', label='V_x')
    axs[0].plot([state[3] for state in state_history], 'g', label='V_y')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('UAV Velocity')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot([state[4] for state in state_history], 'g')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Yaw (radians)')
    axs[1].set_title('UAV Yaw')
    axs[1].grid()

    axs[2].plot([input[0] for input in input_history], 'b', label='Acceleration_x')
    axs[2].plot([input[1] for input in input_history], 'g', label='Acceleration_y')
    axs[2].plot([input[2] for input in input_history], 'r', label='Yaw_rate')
    axs[2].plot([input[3] for input in input_history], 'c', label='Speed')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Input Value')
    axs[2].legend()
    axs[2].set_title('Control Inputs')
    axs[2].grid()

    axs[3].plot([state[5] for state in state_history], 'purple')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Theta')
    axs[3].set_title('UAV Theta (Progress Along Path)')
    axs[3].grid()

    axs[4].plot(cost_history, 'orange')
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Cost')
    axs[4].set_title('MPC Cost')
    axs[4].grid()

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
                vline.set_xdata(closest_index)
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

def animate_horizons(horizons, plane_states, input_history, cost_history, N_horizon, max_simulation_time, horizon_time, sim_dt, path_points=None, interval=100, save_animation=False):
    fig = plt.figure(figsize=(20, 25))
    
    main_ax = plt.subplot2grid((5, 6), (0, 0), rowspan=5, colspan=4)
    axs = [plt.subplot2grid((5, 6), (i, 4), colspan=2) for i in range(5)]
    
    if path_points is not None:
        main_ax.plot(path_points[:, 0], path_points[:, 1], 'k--', alpha=0.5, label='Reference Path')
    
    horizon_lines = LineCollection([], colors='blue', alpha=0.3)
    main_ax.add_collection(horizon_lines)
    
    current_predicted_point, = main_ax.plot([], [], 'bo', markersize=8, label='Predicted Position')
    actual_point, = main_ax.plot([], [], 'ro', markersize=10, label='Actual Position')
    actual_trajectory, = main_ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Actual Trajectory')
    
    all_x = np.concatenate([h[:, 0] for h in horizons] + [[s[0] for s in plane_states]])
    all_y = np.concatenate([h[:, 1] for h in horizons] + [[s[1] for s in plane_states]])
    
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    main_ax.set_xlim(np.min(all_x) - 0.2*x_range, np.max(all_x) + 0.2*x_range)
    main_ax.set_ylim(np.min(all_y) - 0.2*y_range, np.max(all_y) + 0.2*y_range)
    
    main_ax.set_xlabel('X position')
    main_ax.set_ylabel('Y position')
    main_ax.set_title('Evolution of Predicted Horizons and Actual Trajectory')
    main_ax.legend()
    
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
        current_predicted_point.set_data([], [])
        actual_point.set_data([], [])
        actual_trajectory.set_data([], [])
        time_text.set_text('')
        for line in velocity_lines + input_lines + [yaw_line, theta_line, cost_line]:
            line.set_data([], [])
        return [horizon_lines, current_predicted_point, actual_point, actual_trajectory, time_text] + velocity_lines + [yaw_line] + input_lines + [theta_line, cost_line]

    def update(frame):
        current_time = frame * sim_dt
        horizon = horizons[frame]
        actual_state = plane_states[frame]
        
        segments = np.array([horizon[i:i+2, :2] for i in range(min(N_horizon-1, len(horizon)-1))])
        horizon_lines.set_segments(segments)
        
        current_predicted_point.set_data(horizon[0, 0], horizon[0, 1])
        
        actual_point.set_data(actual_state[0], actual_state[1])
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
        
        return [horizon_lines, current_predicted_point, actual_point, actual_trajectory, time_text] + velocity_lines + [yaw_line] + input_lines + [theta_line, cost_line]

    total_frames = len(horizons)
    anim = AnimationFunc(fig, update, frames=total_frames, init_func=init, blit=False, interval=interval)
    
    if save_animation:
        anim.save('horizon_evolution.gif', writer='pillow')
    
    plt.tight_layout()
    return anim
