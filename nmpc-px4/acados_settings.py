from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as cs
import numpy as np
from model.FW_lateral_model import FixedWingLateralModel
from Paths.curve import Path
from Paths.waypoints import path_points

def acados_settings(model, N_horizon, Tf, x0, use_RTI):
    model = FixedWingLateralModel()
    path = Path(path_points)

    ocp = AcadosOcp()
    ocp.model = model.fixed_wing_lateral_model()

    Q_cont = 1.0
    Q_lag = 1.0
    R_1 = 0.1
    R_2 = 0.1
    R_3 = 0.1
    R_4 = 0.1

    ocp.dims.N = N_horizon
    mpc_dt = Tf / N_horizon

    # Increase number of parameters to include cost weights
    ocp.parameter_values = np.zeros((5, 1))
    ocp.cost.cost_type = 'EXTERNAL'

    # States
    I_n = ocp.model.x[0]
    I_e = ocp.model.x[1]

    # Parameters (including reference and weights)
    n_ref = ocp.model.p[2]
    e_ref = ocp.model.p[3]
    phi_ref = ocp.model.p[4]

    # Calculate errors
    eC = cs.sin(phi_ref) * (e_ref - I_e) - cs.cos(phi_ref) * (n_ref - I_n)
    eL = cs.cos(phi_ref) * (e_ref - I_e) + cs.sin(phi_ref) * (n_ref - I_n)

    # Cost function
    c_eC = eC * Q_cont * eC
    c_eL = eL * Q_lag * eL
    c_aX, c_aY = ocp.model.u[0] * R_1 * ocp.model.u[0], ocp.model.u[1] * R_2 * ocp.model.u[1]
    c_yR = ocp.model.u[2] * R_3 * ocp.model.u[2]
    c_vK = -ocp.model.u[3] * R_4  # Small weight to encourage forward motion
    
    ocp.model.cost_expr_ext_cost = c_eC + c_eL + c_vK

    ocp.constraints.lbu = np.array([-0.4, -15.0, -np.pi/3, 0.0])
    ocp.constraints.ubu = np.array([0.4, 15.0, np.pi/3, 10.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.lbx = np.array([15.0, 0.0, -2*np.pi, 0.0])
    ocp.constraints.ubx = np.array([25.0, 0.0, 2*np.pi, path.total_length])
    ocp.constraints.idxbx = np.array([2, 3, 4, 5])

    ocp.constraints.x0 = x0

    # Solver options
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_max_iter = 100
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

    return ocp_solver, acados_integrator, mpc_dt
