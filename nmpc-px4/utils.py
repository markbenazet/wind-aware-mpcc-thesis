import matplotlib.pyplot as plt
import numpy as np

def initialize_plot():
    fig = plt.figure(figsize=(20, 15))
    
    # UAV Trajectory
    ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=4)
    line_uav_traj, = ax1.plot([], [], 'b-', label='UAV Trajectory')
    line_path_points, = ax1.plot([], [], 'g--', label='Path Points')
    line_ref_points, = ax1.plot([], [], 'r.', label='Reference Points')
    arrow_vector = ax1.arrow(0, 0, 0, 0, color='magenta', width=4.0, length_includes_head=True, head_width=4.0)
    ax1.set_xlabel('East')
    ax1.set_ylabel('North')
    ax1.set_title('UAV Trajectory')
    ax1.legend()
    ax1.grid()

    # Right subplots for State Variables and Control Inputs
    axs = [plt.subplot2grid((3, 6), (i, 4), colspan=2) for i in range(3)]

    # Velocity
    axs[0].set_title('UAV Velocity')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Velocity (m/s)')
    line_vx, = axs[0].plot([], [], 'b', label='V_x')
    line_vy, = axs[0].plot([], [], 'g', label='V_y')
    axs[0].legend()
    axs[0].grid()

    # Yaw
    axs[1].set_title('UAV Yaw')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Yaw (radians)')
    line_yaw, = axs[1].plot([], [], 'g')
    axs[1].grid()

    # Control Inputs
    axs[2].set_title('Control Inputs')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Input Value')
    line_accel_x, = axs[2].plot([], [], 'b', label='Acceleration_x')
    line_accel_y, = axs[2].plot([], [], 'g', label='Acceleration_y')
    line_yaw_rate, = axs[2].plot([], [], 'r', label='Yaw_rate')
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.ion()
    plt.show()

    return fig, ax1, axs, line_uav_traj, line_path_points, line_ref_points, arrow_vector, line_vx, line_vy, line_yaw, line_accel_x, line_accel_y, line_yaw_rate

def update_plot(ax1, axs, line_uav_traj, line_path_points, line_ref_points, arrow_vector,
                line_vx, line_vy, line_yaw, line_accel_x, line_accel_y, line_yaw_rate,
                state_history, path_points, reference_history, input_history, vector_p):
    
    line_uav_traj.set_data([state[1] for state in state_history], [state[0] for state in state_history])
    line_path_points.set_data([p[1] for p in path_points], [p[0] for p in path_points])
    line_ref_points.set_data([p[1] for p in reference_history], [p[0] for p in reference_history])

    # Update Velocity
    line_vx.set_data(range(len(state_history)), [state[2] for state in state_history])
    line_vy.set_data(range(len(state_history)), [state[3] for state in state_history])

    # Update Yaw
    line_yaw.set_data(range(len(state_history)), [state[4] for state in state_history])

    # Update Control Inputs
    line_accel_x.set_data(range(len(input_history)), [input[0] for input in input_history])
    line_accel_y.set_data(range(len(input_history)), [input[1] for input in input_history])
    line_yaw_rate.set_data(range(len(input_history)), [input[2] for input in input_history])

    ax1.relim()
    ax1.autoscale_view()

    for ax in axs:
        ax.relim()
        ax.autoscale_view()

    plt.draw()
    plt.pause(0.01)

