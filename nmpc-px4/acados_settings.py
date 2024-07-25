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

    ocp.dims.N = N_horizon
    mpc_dt = Tf / N_horizon

    ocp.parameter_values = np.zeros((5, 1))
    ocp.cost.cost_type = 'EXTERNAL'

    I_n = ocp.model.p[2]
    I_e = ocp.model.p[3]
    Theta = ocp.model.p[4]

    Q_cont = 10.0
    Q_lag = 1.0
    R_1, R_2 = 1.0, 1.0
    R_3 = 1.0
    gamma = 1.0

    n_ref, e_ref = path.evaluate_path(Theta)
    phi = path.get_tangent_angle(Theta)

    # Calculate errors
    eC = -cs.cos(phi) * (I_n - n_ref) + cs.sin(phi) * (I_e - e_ref)
    eL = cs.sin(phi) * (I_n - n_ref) + cs.cos(phi) * (I_e - e_ref)

    c_eC = eC*Q_cont*eC
    c_eL = eL*Q_lag*eL
    c_aX, c_aY = ocp.model.u[0]*R_1*ocp.model.u[0], ocp.model.u[1]*R_2*ocp.model.u[1]
    c_yR = ocp.model.u[2]*R_3*ocp.model.u[2]
    c_vK = -ocp.model.u[3]*gamma
    
    ocp.model.cost_expr_ext_cost = c_eC + c_eL + c_aX + c_aY + c_yR + c_vK

    ocp.constraints.lbu = np.array([-0.4, -15.0, -np.pi/3, 0.0])
    ocp.constraints.ubu = np.array([0.4, 15.0, np.pi/3, 0.001])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.lbx = np.array([15.0, 0.0, -2*np.pi, 0.0])
    ocp.constraints.ubx = np.array([25.0, 0.0, 2*np.pi, -1.0])
    ocp.constraints.idxbx = np.array([2, 3, 4, 5])

    ocp.constraints.x0 = x0

    ocp.solver_options.tf = mpc_dt
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
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
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp_solver, acados_integrator, mpc_dt
