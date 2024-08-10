from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as cs
import numpy as np
from model.FW_lateral_model import FixedWingLateralModel
from Paths.curve import Path
from Paths.waypoints import path_points

def acados_settings(model, N_horizon, Tf, x0, num_laps, use_RTI):
    model = FixedWingLateralModel()
    path = Path(path_points, num_laps)

    ocp = AcadosOcp()
    ocp.model = model.fixed_wing_lateral_model()

    Q_cont = 20.0
    Q_lag = 20.0
    R_1 = 5.0
    R_2 = 1.0
    R_3 = 1.0
    R_4 = 7.0
    R_airspeed = 10.0

    ocp.dims.N = N_horizon
    mpc_dt = Tf / N_horizon

    # Increase number of parameters to include cost weights
    ocp.parameter_values = np.zeros((2,1))
    ocp.cost.cost_type = 'EXTERNAL'

    # States
    I_x = ocp.model.x[0]
    I_y = ocp.model.x[1]
    B_v_x = ocp.model.x[2]
    B_v_y = ocp.model.x[3]
    Theta = ocp.model.x[5]
    
    # Calculate airspeed
    airspeed = cs.sqrt(B_v_x**2 + B_v_y**2)

    x_ref, y_ref = path.evaluate_path(Theta)
    phi_ref = path.get_tangent_angle(Theta)

    # Calculate errors
    eC = cs.sin(phi_ref) * (I_x-x_ref) - cs.cos(phi_ref) * (I_y-y_ref)
    eL = -cs.cos(phi_ref) * (I_x - x_ref) - cs.sin(phi_ref) * (I_y - y_ref)

    V_desired = 20.0
    V_min, V_max = 18.0, 22.0

    # Cost function
    c_eC = eC * Q_cont * eC
    c_eL = eL * Q_lag * eL
    c_aX, c_aY = ocp.model.u[0] * R_1 * ocp.model.u[0], ocp.model.u[1] * R_2 * ocp.model.u[1]
    c_yR = ocp.model.u[2] * R_3 * ocp.model.u[2]
    c_vK = -ocp.model.u[3] * R_4  # Small weight to encourage forward motion
    c_airspeed = R_airspeed * (airspeed - V_desired)**2
    
    ocp.model.cost_expr_ext_cost = c_vK + c_eC + c_aX + c_aY + c_yR + c_eL + c_airspeed
    
    # Control input constraints
    ocp.constraints.lbu = np.array([-0.4, -30.0, -np.pi/3, 0.0])
    ocp.constraints.ubu = np.array([0.4, 30.0, np.pi/3, 50.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # State constraints
    ocp.constraints.lbx = np.array([0.0, 0.0])
    ocp.constraints.ubx = np.array([0.0, path.extended_length])
    ocp.constraints.idxbx = np.array([3, 5])  # y velocity and theta
    ocp.constraints.lbx_0 = np.concatenate([ocp.constraints.lbx, ocp.constraints.lbx])
    ocp.constraints.ubx_0 = np.concatenate([ocp.constraints.ubx, ocp.constraints.ubx])
    ocp.constraints.idxbx_0 = np.concatenate([ocp.constraints.idxbx, ocp.constraints.idxbx + 6])

    # Nonlinear constraints (including airspeed)
    ocp.model.con_h_expr = airspeed
    ocp.model.con_h_expr_0 = airspeed

    # Set the bounds for the hard constraint
    ocp.constraints.lh = np.array([V_min])
    ocp.constraints.uh = np.array([V_max])
    ocp.constraints.lh_0 = ocp.constraints.lh
    ocp.constraints.uh_0 = ocp.constraints.uh

    # Add soft constraints on desired airspeed
    ocp.constraints.lsh = np.array([V_desired - V_min])  # Slack for lower bound
    ocp.constraints.ush = np.array([V_max - V_desired])  # Slack for upper bound
    ocp.constraints.idxsh = np.array([0])  # Apply slack to the airspeed constraint
    ocp.constraints.lsh_0 = ocp.constraints.lsh
    ocp.constraints.ush_0 = ocp.constraints.ush
    ocp.constraints.idxsh_0 = ocp.constraints.idxsh

    # Penalties for soft constraints
    slack_penalty = 1000.0
    ocp.cost.Zl = np.array([slack_penalty])
    ocp.cost.Zu = np.array([slack_penalty])
    ocp.cost.zl = np.array([slack_penalty])
    ocp.cost.zu = np.array([slack_penalty])
    ocp.cost.Zl_0 = ocp.cost.Zl
    ocp.cost.Zu_0 = ocp.cost.Zu
    ocp.cost.zl_0 = ocp.cost.zl
    ocp.cost.zu_0 = ocp.cost.zu

    ocp.constraints.x0 = x0

    # Solver options
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.nlp_solver_max_iter = 500
    ocp.solver_options.tol = 1e-4

    if use_RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.qp_solver_cond_N = N_horizon

    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp_solver, acados_integrator, mpc_dt, ocp.constraints
