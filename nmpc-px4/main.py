from Paths import path_manager
from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate path points from waypoints
    path_points = path_manager.generate_path_from_waypoints()

    # Create model instance
    model = FixedWingLateralModel()

    # Initialize MPC solver
    N_horizon = 20  # Example prediction horizon (adjust as needed)
    Tf = 5.0  # Example prediction horizon duration (adjust as needed)

    # Initial state for MPC solver (adjust as needed)
    x0 = np.array([0.0, 0.0, 10.0, 0.0])  # Example initial state (x, y, V, yaw)

    # Correct way to pass x0 as an argument with its name specified
    ocp_solver, acados_integrator = acados_settings(model, N_horizon, Tf, path_points, x0, use_RTI=False)

    # Threshold distance to switch to the next point (adjust as needed)
    threshold_distance = 5.0  # Example threshold distance

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    output_history = []

    current_state = x0

    # Create figure for plotting
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    # Main control loop
    i = 0
    while i < len(path_points):
        current_point = np.array(path_points[i])

        # Print the current waypoint index
        #print(f"Current state: {current_state}")
        print(f"Current waypoint: {current_point}")

        # Update MPC reference
        ocp_solver.set(0, 'lbx', current_state)
        ocp_solver.set(0, 'ubx', current_state)
        ocp_solver.set(0, 'yref', np.concatenate((current_point, np.zeros(5))))

        # Solve MPC problem
        status = ocp_solver.solve()
        if status != 0:
            print(f"acados returned status {status} in closed loop iteration {i}.")

        # Get control inputs from MPC solver
        u_opt = ocp_solver.get(0, 'u')

        # Store current state, input, and output for debugging
        state_history.append(current_state.copy())
        input_history.append(u_opt.copy())
        output_history.append(current_point.copy())

        # Update current_state based on dynamics model
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", u_opt)
        acados_integrator.solve()
        current_state = acados_integrator.get("x")

        # Check if the current state is close enough to the current reference point
        distance = np.linalg.norm(current_state[:2] - current_point[:2])
        if distance < threshold_distance:
            i += 1  # Switch to the next point

        # Plotting in real-time
        axs[0].cla()
        axs[0].plot(current_state[0], current_state[1], 'bo')  # Plot current UAV position
        axs[0].plot([p[0] for p in path_points], [p[1] for p in path_points], 'g--')  # Plot path
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('UAV Trajectory')

        axs[1].cla()
        axs[1].plot([state[2] for state in state_history], 'r')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Velocity')

        axs[2].cla()
        axs[2].plot([state[3] for state in state_history], 'g')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Yaw')

        axs[3].cla()
        if input_history:
            inputs = np.array(input_history)
            axs[3].plot(inputs[:, 0], 'b', label='Ax')
            axs[3].plot(inputs[:, 1], 'r', label='Ay')
            axs[3].plot(inputs[:, 2], 'g', label='Yaw rate')
            axs[3].set_xlabel('Time Step')
            axs[3].set_ylabel('Input Value')
            axs[3].legend()
            axs[3].set_title('Control Inputs')

        plt.pause(0.01)

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()

    # Optionally, save the state, input, and output histories for further analysis
    np.save("state_history.npy", state_history)
    np.save("input_history.npy", input_history)
    np.save("output_history.npy", np.array(output_history))

if __name__ == "__main__":
    main()