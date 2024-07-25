from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points

def call_mpcc(x0, ocp_solver, acados_integrator, params, u_opt):
    # From an initial state x0 computes the optimal control input u_opt and the corresponding state trajectory 
    return x, u # state and input for all the horizon (matrix)

def warm_start(x0, ocp_solver, u_opt, N_horizon):
    # Warm start the solver to generate the first trajectory point
    iter = 30
    optimal_x = np.zeros((N_horizon, 6))
    optimal_u = np.zeros((N_horizon, 4))
    for idx in range(iter):
        #call_mpcc (with optimal trajectory) and save optimal trajectory

    return optimal_x, optimal_u

def main():
    # Create model instance
    model = FixedWingLateralModel()
    path = Path(path_points)
    # Initialize MPC solver
    N_horizon = 10
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

    params = np.zeros(5)  # Adjust size if needed

    current_state = x0.copy()  # Initialize current_state
    u_opt = np.zeros(4)  # Initialize u_opt with the correct size

    optimal_x, optimal_u = warm_start(x0, ocp_solver, u_opt)

    previous_x, previous_u = optimal_x, optimal_u

    while simulation_time < max_simulation_time:
        
        x_opt, u_opt =  call_mpcc(previous_x, previous_u, ocp_solver, acados_integrator, params) 

        previous_x = x_opt
        previous_u = u_opt

        apply_control_input = u_opt[0]
        
        # simulate fixed wing
        current_state = acados_integrator.simulate(current_state, apply_control_input, mpc_dt)

    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()
