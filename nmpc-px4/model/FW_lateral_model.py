from acados_template import AcadosModel
import casadi as cs

class FixedWingLateralModel():
    def __init__(self):
        
        self.model_name = 'fixed_wing_lateral_model'

        #constants
        self.gravity = 9.81
        # self.mass = 1.0
        # self.max_thrust = 1.0

    def fixed_wing_lateral_model(self) -> AcadosModel:

        # State variables
        I_n = cs.MX.sym('e')
        I_e = cs.MX.sym('n')
        B_V = cs.MX.sym('V')
        I_yaw = cs.MX.sym('yaw')

        # Input variables
        B_a = cs.MX.sym('a')
        I_roll= cs.MX.sym('roll')

        # State and Input vectors
        states = cs.vertcat(I_e, I_n, B_V, I_yaw)
        controls = cs.vertcat(B_a, I_roll)

        # Define the dynamics equations
        dn_dt = B_V * cs.cos(I_yaw)
        de_dt = B_V * cs.sin(I_yaw)
        dV_dt = B_a
        dyaw_dt = self.gravity * cs.tan(I_roll) / B_V

        # Concatenate the state derivatives
        state_derivatives = cs.vertcat(dn_dt, de_dt, dV_dt, dyaw_dt)

        # AcadosModel object
        model = AcadosModel()
        model.f_expl_expr = state_derivatives
        model.x = states
        model.u = controls
        model.name = self.model_name

        return model
