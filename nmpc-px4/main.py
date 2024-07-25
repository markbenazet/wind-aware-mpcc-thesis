from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points

def initialize_trajectory(ocp_solver, path, N_horizon, x0, mpc_dt):
    theta_old = x0[5] * np.ones((N_horizon,))
    x_current = np.tile(x0, (N_horizon, 1))

    for _ in range(30):  # Number of iterations for initialization
        for stageidx in range(N_horizon):
            n_ref, e_ref = path.evaluate_path(theta_old[stageidx])
            phi = path.get_tangent_angle(theta_old[stageidx])
            p_val = np.array([0.0, 0.0, n_ref, e_ref, theta_old[stageidx]])
            ocp_solver.set(stageidx, "p", p_val)
            ocp_solver.set(stageidx, "x", x_current[stageidx])

        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        status = ocp_solver.solve()
        if status != 0:
            print(f"Initialization solve failed with status {status}")
            break

        for idx_sol in range(N_horizon):
            x_current[idx_sol] = ocp_solver.get(idx_sol, "x")

        theta_current = x_current[:, 5]
        theta_diff = np.sum(np.abs(theta_current - theta_old))
        print(f"Theta init difference: {theta_diff}")
        theta_old = theta_current

    return x_current, theta_current

def main():
    # Create model instance
    model = FixedWingLateralModel()
    path = Path(path_points)
    # Initialize MPC solver
    N_horizon = 20
    Tf = 1.0
    # Initial state for MPC solver
    x0 = np.array([0.0, -20.0, 20.0, 0.0, 0.0, 0.0])  # initial state (x, y, V, yaw)
    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)
    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    # Generate reference points using Path class
    reference_history = path.spline_points
    simulation_time = 0
    max_simulation_time = 40.0

    params = np.zeros(5)  # Adjust size if needed
    params[0:5] = [0.0, 0.0, x0[0], x0[1], x0[5]]  # Wind parameters
    x_warm, theta_vals = initialize_trajectory(ocp_solver, path, N_horizon, x0, mpc_dt)
    u_warm = np.zeros((N_horizon, 4))
    current_state = x0.copy()

    while simulation_time < max_simulation_time:
        # Set initial state constraint
        wrapped_current_state = current_state.copy()
        wrapped_current_state[4] = model.np_wrap_angle(wrapped_current_state[4])
        ocp_solver.set(0, 'lbx', wrapped_current_state)
        ocp_solver.set(0, 'ubx', wrapped_current_state)

        # Warm start initialization
        for i in range(N_horizon):
            n_ref, e_ref = path.evaluate_path(theta_vals[i])
            phi = path.get_tangent_angle(theta_vals[i])
            p_val = np.array([0.0, 0.0, n_ref, e_ref, theta_vals[i]])
            ocp_solver.set(i, "p", p_val)
            ocp_solver.set(i, "x", x_warm[i])
            ocp_solver.set(i, "u", u_warm[i])

        # Solve the optimization problem
        status = ocp_solver.solve()

        if status != 0:
            print(f"acados returned status {status} in closed loop iteration at time {simulation_time:.2f}.")
            break

        # Extract the optimized trajectory and inputs
        for i in range(N_horizon):
            x_warm[i] = ocp_solver.get(i, 'x')
            u_warm[i] = ocp_solver.get(i, 'u')

        # Use the first control input to update the state
        acados_integrator.set('x', x_warm[0])
        acados_integrator.set('u', u_warm[0])
        acados_integrator.solve()
        current_state = acados_integrator.get('x')
        current_state[4] = model.np_wrap_angle(current_state[4])

        # Store results and update time
        state_history.append(current_state.copy())
        input_history.append(u_warm[0].copy())
        simulation_time += mpc_dt

        # Shift warm start arrays for next iteration
        x_warm = np.roll(x_warm, -1, axis=0)
        u_warm = np.roll(u_warm, -1, axis=0)
        theta_vals = np.roll(theta_vals, -1)

        # Estimate new final state and input (simple extrapolation)
        x_warm[-1] = x_warm[-2] + (x_warm[-2] - x_warm[-3])
        u_warm[-1] = u_warm[-2]
        theta_vals[-1] = theta_vals[-2] + (theta_vals[-2] - theta_vals[-3])

        # Print progress
        n_ref, e_ref = path.evaluate_path(current_state[5])
        phi = path.get_tangent_angle(current_state[5])
        print(f"State at time {simulation_time:.2f}: theta={current_state[5]:.2f}, n_ref={n_ref:.2f}, e_ref={e_ref:.2f}, phi={phi:.2f}, Objective={ocp_solver.get_cost()}, Current State={current_state[:2]}")

        simulation_time += mpc_dt

    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()
