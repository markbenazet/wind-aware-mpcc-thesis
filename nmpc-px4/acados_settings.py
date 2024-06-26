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

def acados_settings(model, N_horizon, Tf, path_points, x0,use_RTI=False):
    # Create an instance of the FixedWingLateralModel
    model = FixedWingLateralModel()

    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Define model for ocp
    ocp.model = model.fixed_wing_lateral_model()

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Set dimensions
    ocp.dims.N = N_horizon

    # Set cost
    Q_mat = np.diag([100, 100, 1e-8, 1e-8])  # Adjust as needed
    R_mat = np.eye(nu)

    unscale = N_horizon / Tf

    # Initial reference based on the first point in path_points
    x_ref, y_ref = path_points[0]

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.array([x_ref, y_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.W = unscale * scipy.linalg.block_diag(Q_mat, R_mat)

    # Set constraints
    # ocp.constraints.lbu = np.array([9.81, -19.62, -2.0])
    # ocp.constraints.ubu = np.array([19.62, 19.62, 2.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2])

    # ocp.constraints.lbx = np.array([-1.0e19, -1.0e19, 0.0, -1.0e19])
    # ocp.constraints.ubx = np.array([1.0e19, 1.0e19, 40, 1.0e19])
    # ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = x0

    # Set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'

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

    return ocp_solver, acados_integrator

