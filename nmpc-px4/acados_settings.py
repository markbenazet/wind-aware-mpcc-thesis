from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as cs
import numpy as np
from model.FW_lateral_model import FixedWingLateralModel
from Paths.curve import Path

def acados_settings(model, N_horizon, Tf, x0, use_RTI):
    model = FixedWingLateralModel()
    path = Path()

    ocp = AcadosOcp()
    ocp.model = model.fixed_wing_lateral_model()

    ocp.dims.N = N_horizon
    mpc_dt = Tf / N_horizon

    ocp.parameter_values = np.zeros((2, 1))
    ocp.cost.cost_type = 'EXTERNAL'

    I_n = ocp.model.x[0]
    I_e = ocp.model.x[1]
    Theta = ocp.model.x[5]

    ocp.cost.yref = np.zeros(4)

    Q_cont = 1.0
    Q_lag = 1.0
    Q = np.diag([Q_cont, Q_lag])
    R = np.diag([2.0, 2.0, 5.0])
    gamma = 1.0

    n_ref, e_ref = path.evaluate_path_symbolic(Theta)
    phi = path.get_tangent_angle_symbolic(Theta)

    e_c = -cs.cos(phi) * (I_n - n_ref) + cs.sin(phi) * (I_e - e_ref)
    e_l = -cs.sin(phi) * (I_n - n_ref) - cs.cos(phi) * (I_e - e_ref)

    ocp.model.cost_expr_ext_cost = (
        cs.vertcat(e_c, e_l).T @ Q @ cs.vertcat(e_c, e_l) +
        cs.vertcat(ocp.model.u[:3]).T @ R @ cs.vertcat(ocp.model.u[:3]) -
        gamma * ocp.model.u[3] * mpc_dt
    )

    ocp.constraints.lbu = np.array([-0.4, -15.0, -np.pi/3])
    ocp.constraints.ubu = np.array([0.4, 15.0, np.pi/3])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.lbx = np.array([15.0, 0.0, -2*np.pi])
    ocp.constraints.ubx = np.array([25.0, 0.0, 2*np.pi])
    ocp.constraints.idxbx = np.array([2, 3, 4])

    ocp.constraints.x0 = x0

    ocp.solver_options.tf = mpc_dt
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_max_iter = 100

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
