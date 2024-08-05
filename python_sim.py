import numpy as np
import matplotlib.pyplot as plt
import utils as u

def warm_start(x0, ocp_solver, N_horizon, path, model, params, max_iter=30, cost_threshold=1e-3):
    optimal_x = np.tile(x0, (N_horizon, 1))
    optimal_u = np.zeros((N_horizon, 4))
    prev_cost = float('inf')
    optimal_x_history = []

    
    
    for idx in range(max_iter):
        # Use the existing call_mpcc function
        new_x, new_u = call_mpcc(optimal_x, optimal_u, ocp_solver, x0, params, N_horizon, model)
        
        
        optimal_x, optimal_u = new_x, new_u
        optimal_x_history.append(optimal_x)
        
        current_cost = ocp_solver.get_cost()
        # print(f"Iteration {idx+1}, Cost: {current_cost:.6f}, Change: {cost_change:.6f}")

        if abs(current_cost - prev_cost) < cost_threshold:
            break
        prev_cost = current_cost
    
    u.plot_warm_start(optimal_x_history, path.spline_points, N_horizon, max_iterations=idx)

    return optimal_x, optimal_u

def call_mpcc(previous_x, previous_u, ocp_solver, current_state, params, N_horizon, model):
    # From an initial state x0 computes the optimal control input u_opt and the corresponding state trajectory 

    # Set initial state
    ocp_solver.set(0, 'lbx', current_state)
    ocp_solver.set(0, 'ubx', current_state)

    # Update MPC reference for all prediction steps
    for idx in range(N_horizon):
        ocp_solver.set(idx, 'x', previous_x[idx, :])
        ocp_solver.set(idx, 'u', previous_u[idx, :])
        ocp_solver.set(idx, 'p', params)


    # print ("current_state", current_state)

    # Solve MPC problem
    status = ocp_solver.solve()
    # if status != 0:
        # print("acados returned status {0}".format(status))

    # Initialize matrices to store state and control trajectories
    X = np.zeros((N_horizon, previous_x.shape[1]))  # Assuming previous_x.shape[1] is the state dimension
    U = np.zeros((N_horizon, previous_u.shape[1]))  # Assuming previous_u.shape[1] is the control dimension

    # Retrieve the state and control trajectories from the solver
    for i in range(N_horizon):
        X[i, :] = ocp_solver.get(i, 'x')
        U[i, :] = ocp_solver.get(i, 'u')

    # print("Cost value: ", ocp_solver.get_cost())

    return X, U  # state and input for all the horizon (matrix)

def interpolate_horizon(x_opt, u_opt, dt, model):
    second_last_state = x_opt[-2]
    last_state = x_opt[-1]
    
    state_delta = (last_state - second_last_state) / dt
    
    next_state = last_state + state_delta * dt
    
    shifted_x = np.vstack((x_opt[1:], next_state))
    shifted_x[-1, 4] = model.np_wrap_angle(shifted_x[-1, 4])
    
    shifted_u = np.vstack((u_opt[1:], u_opt[-1]))
    
    return shifted_x, shifted_u