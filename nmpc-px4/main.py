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

    print("Cost value: ", ocp_solver.get_cost())

    return X, U  # state and input for all the horizon (matrix)



def warm_start(x0, ocp_solver, N_horizon, params, model,max_iter=30, cost_threshold=1e-6):
    optimal_x = np.zeros((N_horizon, 6))
    optimal_u = np.zeros((N_horizon, 4))
    prev_cost = float('inf')
    
    for idx in range(max_iter):
        new_x, new_u = call_mpcc(optimal_x, optimal_u, ocp_solver, x0, params, N_horizon, model)
        optimal_x, optimal_u = new_x, new_u
        
        current_cost = ocp_solver.get_cost()
        print(f"Iteration {idx+1}, Cost: {current_cost}")
        
        if abs(prev_cost - current_cost) < cost_threshold:
            print(f"Converged after {idx+1} iterations")
            break
        
        prev_cost = current_cost

    return optimal_x, optimal_u

def main():
    # Create model instance
    model = FixedWingLateralModel()
    path = Path(path_points)
    # Initialize MPC solver
    N_horizon = 10
    Tf = 1.0
    # Initial state for MPC solver
    x0 = np.array([0.0, -10.0, 20.0, 0.0, -np.pi, 0.0])  # initial state (x, y, V, yaw)
    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)
    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    # Generate reference points using Path class
    reference_history = path.spline_points
    simulation_time = 0
    max_simulation_time = 40.0

    params = np.zeros(2)  # Adjust size if needed

    current_state = x0.copy()  # Initialize current_state

    optimal_x, optimal_u = warm_start(x0, ocp_solver, N_horizon, params, model)
    print ("Warm start done")
    print("Optimal x: ", optimal_x, "Optimal u: ", optimal_u)

    import matplotlib.pyplot as plt

    # Plot the optimal trajectory of the warm start
    plt.figure()
    plt.plot(optimal_x[:, 1], optimal_x[:, 0], 'b-', label='Optimal Trajectory')
    plt.plot([p[0] for p in reference_history], [p[1] for p in reference_history], 'r.', label='Reference Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimal Trajectory of Warm Start')
    plt.legend()
    plt.grid(True)
    plt.show()

    previous_x, previous_u = optimal_x, optimal_u

    while simulation_time < max_simulation_time:
        
        x_opt, u_opt =  call_mpcc(previous_x, previous_u, ocp_solver, current_state, params, N_horizon, model)

        previous_x = x_opt
        previous_u = u_opt

        apply_control_input = u_opt[0,:]
        
        # simulate fixed wing
        current_state = acados_integrator.simulate(current_state, apply_control_input, mpc_dt)

        state_history.append(current_state)
        input_history.append(apply_control_input)
        simulation_time += mpc_dt

    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()
