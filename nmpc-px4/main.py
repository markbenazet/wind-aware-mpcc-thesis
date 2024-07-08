from Paths import path_manager
from Paths.path_manager import path_manager
from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate path points from waypoints
    path_points = path_manager.generate_path_from_waypoints()

    # Create model instance
    model = FixedWingLateralModel()

    # Initialize MPC solver
    N_horizon = 80
    Tf = 8.0  
    desired_velocity = np.array([20.0, 0.0])
    lookahead_distance = 20.0

    # Initial state for MPC solver 
    x0 = np.array([0.0, -100.0, 20.0, 0.0, -1/2*np.pi])  # initial state (x, y, V, yaw)

    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, path_points, x0, use_RTI=True)

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    reference_history = []

    current_state = x0

    i = 0
    simulation_time = 0
    max_simulation_time = 130.0
    dt = mpc_dt

    while simulation_time < max_simulation_time:
        
        current_position = current_state[:2]
        reference_point = path_manager.get_reference_point(current_position, lookahead_distance)
        lookahead_distance = np.linalg.norm(reference_point-current_position)*0.5

        # Get tangent of the path at the reference position
        tangent_vector = path_manager.get_path_tangent(reference_point)

        n_ref, e_ref = reference_point[0], reference_point[1]

         # Normalize the tangent vector
        tangent_norm = np.linalg.norm(tangent_vector)
        Td_n, Td_e = tangent_vector / tangent_norm if tangent_norm != 0 else (1.0, 0.0)

        # Create parameter vector
        params = np.zeros(8)
        params[:2] = [0.0, 0.0]  # Set wind parameters
        params[2:4] = reference_point
        params[4:6] = [Td_n, Td_e]
        params[6:8] = desired_velocity
        # print(f"Iteration {i}: params = {params}")

        # Update MPC reference for all prediction steps
        for i in range(N_horizon):
            ocp_solver.set(i, 'p', params)
            ocp_solver.set(i, 'yref', np.zeros(7))  # [et, e_chi, e_Vx, e_Vy, B_a_x, B_a_y, I_yaw_rate]

        # Set the initial state constraint
        wrapped_current_state = current_state.copy()
        wrapped_current_state[4] = model.np_wrap_angle(wrapped_current_state[4])
        ocp_solver.set(0, 'lbx', wrapped_current_state)
        ocp_solver.set(0, 'ubx', wrapped_current_state)

        # Solve MPC problem
        status = ocp_solver.solve()
        if status != 0:
            print(f"acados returned status {status} in closed loop iteration at time {simulation_time:.2f}.")
            print(f"Current state: {current_state}")
            break  # Exit the loop if solver fails

        # Get control inputs from MPC solver
        u_opt = ocp_solver.get(0, 'u')

        # Store current state, input, and output for final plotting
        state_to_save = current_state.copy()
        state_history.append(state_to_save)
        input_history.append(u_opt.copy())
        reference_history.append(reference_point.copy())

        # Update current_state based on dynamics model
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", u_opt)
        acados_integrator.solve()
        current_state = acados_integrator.get("x")

        current_state[4] = model.np_wrap_angle(current_state[4])

        # Debug print
        # print(f"Current state: {current_state}")
        # print(f"Reference point: {reference_point}")
        # print(f"Next point: {next_point}")
        # print(f"Yaw reference: {yaw_reference}")
        # print(f"Full reference: {full_reference}")
        # print(f"optimal input: {u_opt}")
        # print(f"Simulation time: {simulation_time:.2f} / {max_simulation_time:.2f}")
        # print(f"Cost function value: {ocp_solver.get_cost()}")

        # Update simulation time
        simulation_time += dt

    u.plot_uav_trajectory_and_state(state_history, path_points, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()