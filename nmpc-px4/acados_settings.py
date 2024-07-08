#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as cs
import numpy as np
from model.FW_lateral_model import FixedWingLateralModel 
import scipy.linalg

def acados_settings(model, N_horizon, Tf, path_points, x0,use_RTI=True):
    # Create an instance of the FixedWingLateralModel
    model = FixedWingLateralModel()

    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Define model for ocp
    ocp.model = model.fixed_wing_lateral_model()

    ocp.dims.N = N_horizon
   
    unscale = N_horizon / Tf

    mpc_dt = Tf / N_horizon

    # Initialize parameters
    ocp.parameter_values = np.zeros((8, 1))

    ocp.cost.cost_type = 'NONLINEAR_LS'

    # Cost function
    I_n = ocp.model.x[0]
    I_e = ocp.model.x[1]
    B_v_x = ocp.model.x[2]
    B_v_y = ocp.model.x[3]
    I_yaw = ocp.model.x[4]
    wrapped_yaw = model.cs_wrap_angle(I_yaw)

    # Define parameters
    w_n = ocp.model.p[0]
    w_e = ocp.model.p[1]
    n_ref = ocp.model.p[2]
    e_ref = ocp.model.p[3]
    Td_n = ocp.model.p[4]
    Td_e = ocp.model.p[5]
    vel_ref = ocp.model.p[6:8]

    et = (n_ref - I_n)*Td_e + (e_ref - I_e)*Td_n
    chi = wrapped_yaw + cs.atan2(B_v_y, B_v_x)  # course angle
    chi_ref = cs.atan2(Td_e, Td_n)
    e_chi = chi_ref - chi

    e_Vx = vel_ref[0] - B_v_x
    e_Vy = vel_ref[1] - B_v_y

    # Define cost function
    ocp.model.cost_y_expr = cs.vertcat(et, e_chi, e_Vx, e_Vy, ocp.model.u)

    # # Print y_expr on every iteration
    # ocp.solver_options.print_level = 2
    
    # Set initial reference. We'll update this in the MPC loop
    ocp.cost.yref = np.zeros(7) 

    # Weights
    Q_mat = np.diag([2.0, 1.0, 0.1, 0.1])  
    R_mat = np.diag([1e-2, 1e-2, 1e-1])
    ocp.cost.W = unscale * scipy.linalg.block_diag(Q_mat, R_mat)

    # Set constraints
    ocp.constraints.lbu = np.array([-0.4, -15.0, -np.pi/3]) 
    ocp.constraints.ubu = np.array([0.4, 15.0, np.pi/3])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # Set state constraints for all time steps
    ocp.constraints.lbx = np.array([-1.0e19, -1.0e19, 15.0, 0.0, -np.pi])
    ocp.constraints.ubx = np.array([1.0e19, 1.0e19, 25.0, 0.0, np.pi])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = x0

    # Set options
    ocp.solver_options.tf = mpc_dt
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_max_iter = 200

    if use_RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # Set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp_solver, acados_integrator, mpc_dt

