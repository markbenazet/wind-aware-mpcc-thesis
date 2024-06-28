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
    N_horizon = 5
    Tf = 1.0  
    desired_velocity = 25.0

    # Initial state for MPC solver 
    x0 = np.array([-20.0, -20.0, 20.0, 0.0])  # initial state (x, y, V, yaw)

    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, path_points, x0, use_RTI=False)

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    output_history = []

    current_state = x0

    i = 0
    simulation_time = 0
    max_simulation_time = 60.0
    dt = mpc_dt

    while simulation_time < max_simulation_time:
        
        current_position = current_state[:2]
        reference_point = path_manager.get_reference_point(current_position, 5.0)

        # Get the next reference point for yaw calculation
        next_point = path_manager.get_reference_point(reference_point, 5.0)

        # Calculate yaw reference
        delta = next_point - reference_point
        yaw_reference = np.arctan2(delta[1], delta[0])

        # Update MPC reference
        full_reference = np.zeros(6) 
        full_reference[:2] = reference_point 
        full_reference[2] = desired_velocity  
        full_reference[3] = yaw_reference

        # Update MPC reference for all prediction steps
        for i in range(N_horizon):
            ocp_solver.set(i, 'yref', full_reference)

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
        state_history.append(current_state.copy())
        input_history.append(u_opt.copy())
        output_history.append(current_position.copy())

        # Update current_state based on dynamics model
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", u_opt)
        acados_integrator.solve()
        current_state = acados_integrator.get("x")

        # Debug print
        print(f"Current position: {current_position}")
        print(f"Reference point: {reference_point}")
        print(f"Next point: {next_point}")
        print(f"Yaw reference: {yaw_reference}")
        print(f"Full reference: {full_reference}")

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
    axs[3].plot([input[0] for input in input_history], 'b', label='Acceleration')
    axs[3].plot([input[1] for input in input_history], 'r', label='Roll')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Input Value')
    axs[3].legend()
    axs[3].set_title('Control Inputs')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()