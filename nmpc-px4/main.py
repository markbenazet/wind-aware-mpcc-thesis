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
    N_horizon = 20
    Tf = 1.0  

    # Initial state for MPC solver 
    x0 = np.array([-20.0, -20.0, 20.0, 0.0])  # initial state (x, y, V, yaw)

    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, path_points, x0, use_RTI=False)

    # Threshold distance to switch to the next point 
    threshold_distance = 7.0  # threshold distance

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    output_history = []

    current_state = x0

    i = 0
    simulation_time = 0
    max_simulation_time = 10.0
    dt = mpc_dt

    while i < len(path_points) and simulation_time < max_simulation_time:
        current_point = np.array(path_points[i])

        # Update MPC reference
        ocp_solver.set(0, 'lbx', current_state)
        ocp_solver.set(0, 'ubx', current_state)
        ocp_solver.set(0, 'yref', np.concatenate((current_point, np.zeros(4))))

        # Solve MPC problem
        status = ocp_solver.solve()
        if status != 0:
            print(f"acados returned status {status} in closed loop iteration {i}.")

        # Get control inputs from MPC solver
        u_opt = ocp_solver.get(0, 'u')

        # Store current state, input, and output for final plotting
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

        # Update simulation time
        simulation_time += dt

    # Plotting after the simulation is complete
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Plot UAV Trajectory
    axs[0].plot([state[1] for state in state_history], [state[0] for state in state_history], 'b-')
    axs[0].plot([p[1] for p in path_points], [p[0] for p in path_points], 'g--')
    axs[0].set_xlabel('East')
    axs[0].set_ylabel('North')
    axs[0].set_title('UAV Trajectory')

    # Plot Velocity
    axs[1].plot([state[2] for state in state_history], 'r')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('UAV Velocity')

    # Plot Yaw
    axs[2].plot([state[3] for state in state_history], 'g')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Yaw (radians)')
    axs[2].set_title('UAV Yaw')

    # Plot Control Inputs
    inputs = np.array(input_history)
    axs[3].plot(range(len(inputs)), inputs[:, 0], 'b', label='Acceleration')
    axs[3].plot(range(len(inputs)), inputs[:, 1], 'g', label='Roll')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Input Value')
    axs[3].legend()
    axs[3].set_title('Control Inputs')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()