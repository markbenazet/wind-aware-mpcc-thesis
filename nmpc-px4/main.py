from Paths import path_manager
from Paths.path_manager import path_manager
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
    N_horizon = 10
    Tf = 1.0  
    desired_velocity = np.array([25.0, 0.0])

    # Initial state for MPC solver 
    x0 = np.array([0.0, 100.0, 25.0, 0.0, np.pi])  # initial state (x, y, V, yaw)

    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, path_points, x0, use_RTI=True)

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    output_history = []
    reference_history = []

    current_state = x0

    i = 0
    simulation_time = 0
    max_simulation_time = 80.0
    dt = mpc_dt

    while simulation_time < max_simulation_time:
        
        current_position = current_state[:2]
        reference_point = path_manager.get_reference_point(current_position, 10.0)

        # # Get the next reference point for yaw calculation
        # next_point = path_manager.get_reference_point(current_position, 20.0)

        # # Calculate yaw reference
        # delta = next_point - reference_point
        # yaw_reference = np.arctan2(delta[1], delta[0])
        # yaw_reference = (yaw_reference + np.pi) % (2 * np.pi) - np.pi

        # Update MPC reference
        full_reference = np.zeros(8) 
        full_reference[:2] = reference_point 
        full_reference[2:4] = desired_velocity  

        p = np.array([0.0, 0.0])

        # Update MPC reference for all prediction steps
        for i in range(N_horizon):
            ocp_solver.set(i, 'yref', full_reference)
            ocp_solver.set(i, 'p', p)

        # Set the initial state constraint
        ocp_solver.set(0, 'lbx', current_state)
        ocp_solver.set(0, 'ubx', current_state)

        # Solve MPC problem
        status = ocp_solver.solve()
        if status != 0:
            print(f"acados returned status {status} in closed loop iteration at time {simulation_time:.2f}.")
            print(f"Current state: {current_state}")
            print(f"Reference point: {reference_point}")
            print(f"Full reference: {full_reference}")
            break  # Exit the loop if solver fails

        # Get control inputs from MPC solver
        u_opt = ocp_solver.get(0, 'u')

        # Store current state, input, and output for final plotting
        state_to_save = current_state.copy()
        state_to_save[4] = model.normalize_angle(state_to_save[4])
        state_history.append(state_to_save)
        input_history.append(u_opt.copy())
        output_history.append(current_position.copy())
        reference_history.append(reference_point.copy())

        # Update current_state based on dynamics model
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", u_opt)
        acados_integrator.solve()
        current_state = acados_integrator.get("x")

        # Debug print
        # print(f"Current state: {current_state}")
        # print(f"Reference point: {reference_point}")
        # print(f"Next point: {next_point}")
        # print(f"Yaw reference: {yaw_reference}")
        # print(f"Full reference: {full_reference}")
        # print(f"optimal input: {u_opt}")
        print(f"Simulation time: {simulation_time:.2f} / {max_simulation_time:.2f}")

        # Update simulation time
        simulation_time += dt

    # Plot UAV Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot([state[1] for state in state_history], [state[0] for state in state_history], 'b-')
    plt.plot([p[1] for p in path_points], [p[0] for p in path_points], 'g--')
    plt.plot([p[1] for p in reference_history], [p[0] for p in reference_history], 'r.',)
    # Add the vector p as an arrow
    plt.arrow(0, 0, 5*p[1], 5*p[0], color='magenta', width=0.5, length_includes_head=True, head_width=2.0)
    plt.xlabel('East')
    plt.ylabel('North')
    plt.title('UAV Trajectory')
    plt.grid()
    plt.show()

    # Plot State Variables and Control Inputs
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Velocity
    axs[0].plot([state[2] for state in state_history], 'b', label='V_x')
    axs[0].plot([state[3] for state in state_history], 'g', label='V_y')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('UAV Velocity')
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

if __name__ == "__main__":
    main()