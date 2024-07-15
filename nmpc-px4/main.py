from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
import utils as u
import numpy as np
from Paths.curve import Path

def main():
    # Create model instance
    model = FixedWingLateralModel()
    path = Path()

    # Initialize MPC solver
    N_horizon = 30
    Tf = 3.0  

    # Initial state for MPC solver 
    x0 = np.array([0.0, -100.0, 20.0, 0.0, 0.0, 0.0])  # initial state (x, y, V, yaw)

    ocp_solver, acados_integrator, mpc_dt = acados_settings(model, N_horizon, Tf, x0, use_RTI=False)

    # Lists to store state and input values for debugging
    state_history = []
    input_history = []
    reference_history = path.get_bspline_curve()
    
    simulation_time = 0
    max_simulation_time = 80.0
    dt = mpc_dt
    params = np.zeros(2)  # Adjust size if needed
    params[0:2] = [0.0, 0.0]  # Wind parameters

    current_state = x0.copy()  # Initialize current_state

    while simulation_time < max_simulation_time:
        # We don't need to set parameters for each step in the horizon
        # The solver will handle the evolution of theta

        # Set the initial state constraint, including the initial theta
        wrapped_current_state = current_state.copy()
        wrapped_current_state[4] = model.np_wrap_angle(wrapped_current_state[4])
        ocp_solver.set(0, 'lbx', wrapped_current_state)
        ocp_solver.set(0, 'ubx', wrapped_current_state)

        ocp_solver.set(0, 'p', params)

        # Solve MPC problem
        status = ocp_solver.solve()
        if status != 0:
            print(f"acados returned status {status} in closed loop iteration at time {simulation_time:.2f}.")
            print(f"Current state: {current_state}")
            break  # Exit the loop if solver fails

        # Get control inputs from MPC solver
        u_opt = ocp_solver.get(0, 'u')

        # Update current_state based on dynamics model
        acados_integrator.set("x", current_state)
        acados_integrator.set("u", u_opt)
        acados_integrator.solve()
        current_state = acados_integrator.get("x")

        current_state[4] = model.np_wrap_angle(current_state[4])

        # Store results and update time
        state_history.append(current_state.copy())
        input_history.append(u_opt.copy())
        simulation_time += dt

        print(f"Simulation time: {simulation_time:.2f} / {max_simulation_time:.2f}")
    
    u.plot_uav_trajectory_and_state(state_history, reference_history, input_history, params[:2])

if __name__ == "__main__":
    main()