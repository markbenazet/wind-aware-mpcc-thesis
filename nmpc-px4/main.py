from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points

def main():
    # Create model instance
    model = FixedWingLateralModel()
    path = Path(path_points)
    # Initialize MPC solver
    N_horizon = 20
    Tf = 1.0
    # Initial state for MPC solver
    x0 = np.array([0.0, -10.0, 20.0, 0.0, 0.0, 0.0])  # initial state (x, y, V, yaw)
    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)
    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    # Generate reference points using Path class
    reference_history = path.spline_points
    simulation_time = 0
    max_simulation_time = 40.0
    dt = mpc_dt
    params = np.zeros(5)  # Adjust size if needed
    params[0:5] = [0.0, 0.0, x0[0], x0[1], x0[5]]  # Wind parameters
    current_state = x0.copy()  # Initialize current_state
    u_opt = np.zeros(4)  # Initialize u_opt with the correct size

    while simulation_time < max_simulation_time:
        step_sol_x = []
        step_sol_u = []
        for idx in range(N_horizon):
            if idx == 0:
                wrapped_current_state = current_state.copy()
                wrapped_current_state[4] = model.np_wrap_angle(wrapped_current_state[4])
                ocp_solver.set(idx, 'lbx', wrapped_current_state)
                ocp_solver.set(idx, 'ubx', wrapped_current_state)
            else:
                ocp_solver.set(idx, 'x', current_state)
                ocp_solver.set(idx, 'u', u_opt)
            ocp_solver.set(idx, 'p', params)
            status = ocp_solver.solve()
            if status != 0:
                print(f"acados returned status {status} at iteration {idx}.")
                break  # Exit the loop if solver fails
            current_state = ocp_solver.get(idx, 'x')
            u_opt = ocp_solver.get(idx, 'u')
            params[2] = current_state[0]
            params[3] = current_state[1]
            params[4] = current_state[5]
            step_sol_x.append(current_state.copy())
            step_sol_u.append(u_opt.copy())

        if status != 0:
            print(f"acados returned status {status} in closed loop iteration at time {simulation_time:.2f}.")
            break  # Exit the loop if solver fails
        # Use the first control input to update the state
        acados_integrator.set('x', step_sol_x[0])
        acados_integrator.set('u', step_sol_u[0])
        acados_integrator.solve()
        current_state = acados_integrator.get('x')
        current_state[4] = model.np_wrap_angle(current_state[4])
        # Store results and update time
        state_history.append(current_state.copy())
        input_history.append(u_opt.copy())
        simulation_time += dt

        n_ref, e_ref = path.evaluate_path(current_state[5])
        phi = path.get_tangent_angle(current_state[5])

        print(f"State at time {simulation_time:.2f}: {current_state[5], n_ref, e_ref, phi}")
        # print(f"Simulation time: {simulation_time:.2f} / {max_simulation_time:.2f}")

    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()
