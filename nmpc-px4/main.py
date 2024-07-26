from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points
import casadi as cs

def call_mpcc(previous_x, previous_u, ocp_solver, current_state, params, N_horizon, model):
    # From an initial state x0 computes the optimal control input u_opt and the corresponding state trajectory 

    # Update MPC reference for all prediction steps
    for idx in range(N_horizon):
        ocp_solver.set(idx, 'x', previous_x[idx, :])
        ocp_solver.set(idx, 'u', previous_u[idx, :])
        ocp_solver.set(idx, 'p', params)

    # Set initial state
    ocp_solver.set(0, 'lbx', current_state)
    ocp_solver.set(0, 'ubx', current_state)

    # Solve MPC problem
    status = ocp_solver.solve()

    # Initialize matrices to store state and control trajectories
    X = np.zeros((N_horizon, previous_x.shape[1]))  # Assuming previous_x.shape[1] is the state dimension
    U = np.zeros((N_horizon, previous_u.shape[1]))  # Assuming previous_u.shape[1] is the control dimension

    # Retrieve the state and control trajectories from the solver
    for i in range(N_horizon):
        X[i, :] = ocp_solver.get(i, 'x')
        U[i, :] = ocp_solver.get(i, 'u')
        X[i, 4] = model.np_wrap_angle(X[i,4])

    # print("Cost value: ", ocp_solver.get_cost())

    return X, U  # state and input for all the horizon (matrix)



def warm_start(x0, ocp_solver, N_horizon, path, model, params, max_iter=30, cost_threshold=1e-4):
    optimal_x = np.zeros((N_horizon, 6))
    optimal_u = np.zeros((N_horizon, 4))
    prev_cost = float('inf')
    optimal_X = np.empty((0, 6))

    # Initialize Theta (path parameter)
    theta_init = path.project_to_path(x0[0], x0[1])
    optimal_x[:, 5] = np.linspace(theta_init, theta_init + path.total_length / 10, N_horizon)
    
    for idx in range(max_iter):
        # Use the existing call_mpcc function
        new_x, new_u = call_mpcc(optimal_x, optimal_u, ocp_solver, x0, params, N_horizon, model)
        
        optimal_x, optimal_u = new_x, new_u
        optimal_X = np.vstack((optimal_X, optimal_x))
        
        current_cost = ocp_solver.get_cost()
        cost_change = abs(current_cost - prev_cost)
        print(f"Iteration {idx+1}, Cost: {current_cost:.6f}, Change: {cost_change:.6f}")
        
        if cost_change < cost_threshold:
            print(f"Warm start converged after {idx+1} iterations")
            break
        
        prev_cost = current_cost
    
    u.plot_warm_start(optimal_X, path.spline_points, N_horizon, max_iterations=idx)

    return optimal_x, optimal_u

def main():
    # Initialize model, solver, and path
    model = FixedWingLateralModel()
    path = Path(path_points)
    N_horizon = 40
    Tf = 8.0
    x0 = np.array([20.0, 20.0, 20.0, 0.0, -np.pi/2, 0.0])
    ocp_solver, acados_integrator, mpc_dt, constraints = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)

    # Initialize histories
    state_history = []
    input_history = []
    horizon_history = []
    simulation_time = 0
    max_simulation_time = 80.0

    params = np.zeros(2)  # Wind parameters
    current_state = x0.copy()

    # Warm start
    optimal_x, optimal_u = warm_start(x0, ocp_solver, N_horizon, path, model, params)
    # Main simulation loop
    while simulation_time < max_simulation_time:
        x_opt, u_opt = call_mpcc(optimal_x, optimal_u, ocp_solver, current_state, params, N_horizon, model)

        # Store the predicted horizon
        horizon_history.append(x_opt)

        apply_control_input = u_opt[0,:]
         
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", apply_control_input)
        acados_integrator.solve()
        new_state = acados_integrator.get("x")

        # Print cost and first state at each iteration
        print(f"Time: {simulation_time:.2f}, Cost: {ocp_solver.get_cost():.2f}")
        print(f"Current state: {current_state}")
        print(f"First predicted state: {x_opt[0]}")
        print(f"Difference: {x_opt[0] - current_state}")
        print("---")

        state_history.append(new_state)
        input_history.append(apply_control_input)
        simulation_time += mpc_dt
        current_state = new_state

        # Shift the horizon
        optimal_x = np.vstack((x_opt[1:], x_opt[-1]))
        optimal_u = np.vstack((u_opt[1:], u_opt[-1]))

        last_theta = optimal_x[-1, 5] + 0.1
        optimal_x[-1, 5] = last_theta
        optimal_x[-1, 0:2] = path.evaluate_path(last_theta)

    # After the simulation, call the plotting functions
    reference_history = path.spline_points
    vector_p = params  # Assuming params represents the wind vector
    # print("state_history", state_history)

    # Animate the UAV trajectory
    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, vector_p)

if __name__ == "__main__":
    main()