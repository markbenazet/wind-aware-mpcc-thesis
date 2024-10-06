from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import matplotlib.pyplot as plt
import utils as u
from python_sim import call_mpcc, warm_start, interpolate_horizon
import numpy as np
from Paths.curve import Path
from Paths.waypoints import path_points
import casadi as cs

def main():
    model = FixedWingLateralModel()
    num_laps = 3
    path = Path(path_points, num_laps)
    N_horizon = 40
    Tf = 8.0
    x0 = np.array([0.0, -100.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    params = np.array([[-14.0], [-14.0], [2.0], [1.0], [0.1], [0.0], [0.0], [0.0]]) 
    x0[5] = 0.0

    ocp_solver, _, mpc_dt,_ = acados_settings(model, N_horizon, Tf, x0, num_laps, use_RTI=False)
    _, fast_acados_integrator, _, _ = acados_settings(model, 10*N_horizon, Tf, x0, num_laps, use_RTI=False)
    
    state_history = []
    state_history.append(x0)
    input_history = []
    horizon_history = []
    sim_input_history = []
    state_solver_history = []
    sim_state_history = []
    cost_history = []
    state_solver_history.append(x0[0:2])
    simulation_time = 0
    max_simulation_time = 120.0

    optimal_x, optimal_u = warm_start(x0, ocp_solver, N_horizon, path, model, params)
    current_state = x0.copy()

    while simulation_time < max_simulation_time or current_state[5] > path.total_length:
        x_opt, u_opt = call_mpcc(optimal_x, optimal_u, ocp_solver, current_state, params, N_horizon, model)

        horizon_history.append(x_opt)
        state_solver_history.append(x_opt[1])

        apply_control_input = u_opt[0,:]
        sim_input_history.append(apply_control_input)
        
        # Simulate dynamics for one MPC step
        for i in range(10):
            new_state = fast_acados_integrator.simulate(current_state, apply_control_input, z=None, xdot=None, p=params)
            state_history.append(new_state)
            input_history.append(apply_control_input)
            current_state = new_state
            if i == 9:
                sim_state_history.append(new_state)
            i+=1

        optimal_x, optimal_u = interpolate_horizon(x_opt, u_opt, mpc_dt, model)

        cost = ocp_solver.get_cost()
        cost_history.append(cost)
        simulation_time += mpc_dt
        params[6:9] = u_opt[0, 0:2].reshape(-1, 1)

    reference_history = path.spline_points
    vector_p = params[0:2]

    # u.plot_acceleration_tracking(state_history, input_history)
    u.plot_uav_trajectory_and_state(sim_state_history, reference_history, horizon_history, sim_input_history, vector_p, cost_history)

    # u.plot_inputs_and_states(sim_state_history, sim_input_history)

    # anim = u.animate_horizons(horizon_history, sim_state_history, sim_input_history, cost_history, 
                        # N_horizon, max_simulation_time, Tf, mpc_dt, vector_p, 
                        # path_points=path.spline_points, interval=100, save_animation=True)
    # plt.show()

if __name__ == "__main__":
    main()