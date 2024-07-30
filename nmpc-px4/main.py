from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import matplotlib.pyplot as plt
import utils as u
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points
import casadi as cs

def interpolate_horizon(x_opt, u_opt, dt, model):
    second_last_state = x_opt[-2]
    last_state = x_opt[-1]
    
    state_delta = (last_state - second_last_state) / dt
    
    next_state = last_state + state_delta * dt
    
    shifted_x = np.vstack((x_opt[1:], next_state))
    shifted_x[-1, 4] = model.np_wrap_angle(shifted_x[-1, 4])
    
    shifted_u = np.vstack((u_opt[1:], u_opt[-1]))
    
    return shifted_x, shifted_u

def call_mpcc(previous_x, previous_u, ocp_solver, current_state, params, N_horizon, model):
    # From an initial state x0 computes the optimal control input u_opt and the corresponding state trajectory 

    # Set initial state
    ocp_solver.set(0, 'lbx', current_state)
    ocp_solver.set(0, 'ubx', current_state)

    # Update MPC reference for all prediction steps
    for idx in range(1,N_horizon):
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
        X[i, 4] = model.np_wrap_angle(X[i,4])

    # print("Cost value: ", ocp_solver.get_cost())

    return X, U  # state and input for all the horizon (matrix)

def warm_start(x0, ocp_solver, N_horizon, path, model, params, max_iter=30, cost_threshold=1e-4):
    optimal_x = np.zeros((N_horizon, 6))
    optimal_u = np.zeros((N_horizon, 4))
    prev_cost = float('inf')
    optimal_x_history = []

    # Initialize Theta (path parameter)
    
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

def main():
    model = FixedWingLateralModel()
    path = Path(path_points)
    N_horizon = 120
    Tf = 6.0
    x0 = np.array([20.0, 50.0, 20.0, 0.0, -np.pi/2, 0.0])
    ocp_solver, acados_integrator, mpc_dt,_ = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)
    Nsim = 200
    simU = np.zeros((Nsim, 4))
    simX = np.zeros((Nsim+1, 6))
    simX[0,:] = x0
    
    state_history = []
    state_history.append(x0)
    input_history = []
    horizon_history = []
    state_solver_history = []
    cost_history = []
    state_solver_history.append(x0[0:2])
    simulation_time = 0
    max_simulation_time = 80.0

    params = np.zeros(2)  # Wind parameters

    for _ in range(5):
        solve = ocp_solver.solve_for_x0(x0_bar = x0)

    current_state = x0.copy()

    for i in range(int(Nsim)):
        # x_opt, u_opt = call_mpcc(optimal_x, optimal_u, ocp_solver, current_state, params, N_horizon, model)

        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i,:])

        # horizon_history.append(x_opt)
        # state_solver_history.append(x_opt[0])

        # apply_control_input = u_opt[1,:]
         
        # new_state = acados_integrator.simulate(current_state, apply_control_input, params, mpc_dt)

        simX[i+1,:] = acados_integrator.simulate(x=simX[i,:], u=simU[i,:])
        simX[i+1,4] = model.np_wrap_angle(simX[i+1,4])
        
        # state_history.append(new_state)
        # input_history.append(apply_control_input)
        # simulation_time += mpc_dt
        # current_state = new_state

        # optimal_x, optimal_u = interpolate_horizon(x_opt, u_opt, mpc_dt, model)

        cost = ocp_solver.get_cost()
        cost_history.append(cost)

    reference_history = path.spline_points
    vector_p = params

    u.plot_uav_trajectory_and_state(simX, reference_history, _, simU, vector_p, cost_history)
    
    # anim = u.animate_horizons(horizon_history,state_history, N_horizon, max_simulation_time, Tf, mpc_dt, path.spline_points, interval=100, save_animation=True)

    plt.show()

if __name__ == "__main__":
    main()