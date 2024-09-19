import numpy as np
import pandas as pd
from acados_settings import acados_settings
from model.FW_lateral_model import FixedWingLateralModel
from python_sim import call_mpcc, warm_start, interpolate_horizon
import utils as u
from Paths.curve import Path
from Paths.waypoints import path_points
import os
import math

def clear_csv_files(results_dir):
    """Clear existing CSV files in the results directory."""
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(results_dir, filename)
            open(file_path, 'w').close()  # This will clear the file contents
    print("Cleared existing CSV files.")

def run_simulation(x0, params, model, path, N_horizon, Tf, num_laps, max_simulation_time):
    ocp_solver, _, mpc_dt, _ = acados_settings(model, N_horizon, Tf, x0, num_laps, use_RTI=False)
    _, fast_acados_integrator, _, _ = acados_settings(model, 10*N_horizon, Tf, x0, num_laps, use_RTI=True)
    
    optimal_x, optimal_u = warm_start(x0, ocp_solver, N_horizon, path, model, params)
    current_state = x0.copy()
    
    state_history = [current_state]
    input_history = []
    cost_history = []
    simulation_time = 0
    
    while simulation_time < max_simulation_time:
        x_opt, u_opt = call_mpcc(optimal_x, optimal_u, ocp_solver, current_state, params, N_horizon, model)
        apply_control_input = u_opt[0,:]
        
        for _ in range(10):
            new_state = fast_acados_integrator.simulate(current_state, apply_control_input, z=None, xdot=None, p=params)
            state_history.append(new_state)
            input_history.append(apply_control_input)
            current_state = new_state
            cost_history.append(ocp_solver.get_cost())
        
        optimal_x, optimal_u = interpolate_horizon(x_opt, u_opt, mpc_dt, model)
        simulation_time += mpc_dt
    
    return np.array(state_history), np.array(input_history), np.array(cost_history)

def calculate_initial_heading(x, y, path_center_x, path_center_y, path_radius):
    # Calculate the vector from the path center to the point
    dx = x - path_center_x
    dy = y - path_center_y
    
    # Calculate the distance from the point to the path center
    distance = np.sqrt(dx**2 + dy**2)
    
    # Calculate the angle from the path center to the point
    angle = np.arctan2(dx, dy)  # Note: y comes first in arctan2

    yaw_angle = -(angle - np.pi/2) % (2 * np.pi)
    if yaw_angle > np.pi:
        yaw_angle -= 2 * np.pi
    
    # Adjust the angle based on whether the point is inside or outside the path
    if distance <= path_radius:
        # If inside or on the path, head outwards
        adjusted_angle = angle
    else:
        # If outside the path, head inwards
        adjusted_angle = angle + np.pi
    
    # Normalize the angle to [-π, π]
    return adjusted_angle

def main():
    results_dir = 'grid_simulation_results'
    os.makedirs(results_dir, exist_ok=True)
    clear_csv_files(results_dir)
    print(f"Results will be saved in: {os.path.abspath(results_dir)}")

    model = FixedWingLateralModel()
    num_laps = 3
    path = Path(path_points, num_laps)
    N_horizon = 40
    Tf = 8.0
    params = np.array([[-14.0], [-14.0], [2.0], [1.0], [0.1]])
    max_simulation_time = 60.0

    # Define the grid
    grid_size = 150.0  # meters between grid points
    grid_extent = 300.0  # meters in each direction from center
    x_range = np.arange(-grid_extent, grid_extent + grid_size, grid_size)
    y_range = np.arange(-grid_extent, grid_extent + grid_size, grid_size)

    all_state_histories = []
    all_input_histories = []
    all_cost_histories = []

    total_simulations = len(x_range) * len(y_range)
    completed_simulations = 0

    for x_start in x_range:
        for y_start in y_range:
            initial_heading = calculate_initial_heading(x_start, y_start, 0.0, 0.0, 200.0)
            x0 = np.array([x_start, y_start, 20.0, 0.0, initial_heading, 0.0, 0.0, 0.0, 0.0])
            x0[5] = path.project_to_path(x0[0], x0[1], x0[5], Tf/N_horizon, x0[2], x0[3], params, initial=True) + 50.0

            state_history, input_history, cost_history = run_simulation(x0, params, model, path, N_horizon, Tf, num_laps,  max_simulation_time)
            
            all_state_histories.append(state_history)
            all_input_histories.append(input_history)
            all_cost_histories.append(cost_history)
            
            completed_simulations += 1
            print(f"Starting point: ({x_start}, {y_start})")
            print(f"Completed simulation {completed_simulations}/{total_simulations}")

    # Save results
    save_results(all_state_histories, all_input_histories, all_cost_histories, x_range, y_range, results_dir)
    u.plot_trajectories()

def save_results(all_state_histories, all_input_histories, all_cost_histories, x_range, y_range, results_dir):
    # Prepare data for state histories
    state_data = []
    state_columns = ['start_x', 'start_y'] + [f"step_{j}_{name}" for j in range(len(all_state_histories[0])) for name in ['x', 'y', 'vx', 'vy', 'yaw', 'theta', 'ay', 'ay_dot', 'ax']]
    
    # Add columns for mean square cost
    state_columns += [f"step_{j}_mean_square_cost" for j in range(len(all_cost_histories[0]))]
    
    for i, (state_history, cost_history) in enumerate(zip(all_state_histories, all_cost_histories)):
        x_start = x_range[i // len(y_range)]
        y_start = y_range[i % len(y_range)]
        row = [x_start, y_start]
        for state in state_history:
            row.extend(state)
        
        # Calculate and add mean square cost for each step
        cumulative_cost = 0
        for j, cost in enumerate(cost_history):
            cumulative_cost += cost
            mean_square_cost = cumulative_cost / (j + 1)
            row.append(mean_square_cost)
        
        state_data.append(row)
    
    # Create state DataFrame
    state_df = pd.DataFrame(state_data, columns=state_columns)
    state_df.to_csv(os.path.join(results_dir, 'state_histories.csv'), index=False)

    # Prepare data for input histories (unchanged)
    input_data = []
    input_columns = ['start_x', 'start_y'] + [f"step_{j}_{name}" for j in range(len(all_input_histories[0])) for name in ['ax', 'ay', 'yaw_rate', 'speed']]
    
    for i, input_history in enumerate(all_input_histories):
        x_start = x_range[i // len(y_range)]
        y_start = y_range[i % len(y_range)]
        row = [x_start, y_start]
        for input_val in input_history:
            row.extend(input_val)
        input_data.append(row)
    
    # Create input DataFrame
    input_df = pd.DataFrame(input_data, columns=input_columns)
    input_df.to_csv(os.path.join(results_dir, 'input_histories.csv'), index=False)

    print(f"Results saved in '{results_dir}' directory.")

if __name__ == "__main__":
    main()