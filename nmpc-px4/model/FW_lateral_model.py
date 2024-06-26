from acados_template import AcadosModel
import casadi as cs

class FixedWingLateralModel():
    def __init__(self):
        
        self.model_name = 'fixed_wing_lateral_model'

        # #constants
        # self.mass = 1.0
        # self.max_thrust = 1.0

    def fixed_wing_lateral_model(self) -> AcadosModel:

        # State variables
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        V = cs.MX.sym('V')
        yaw = cs.MX.sym('yaw')

        # Input variables
        a_x = cs.MX.sym('a_x')
        a_y = cs.MX.sym('a_y')
        yaw_rate = cs.MX.sym('yaw_rate')

        # State and Input vectors
        states = cs.vertcat(x, y, V, yaw)
        controls = cs.vertcat(a_x, a_y, yaw_rate)

        # Define the dynamics equations
        dx_dt = V * cs.cos(yaw)
        dy_dt = V * cs.sin(yaw)
        dV_dt = a_x*cs.cos(yaw) + a_y*cs.sin(yaw)
        dyaw_dt = yaw_rate

        # Concatenate the state derivatives
        state_derivatives = cs.vertcat(dx_dt, dy_dt, dV_dt, dyaw_dt)

        # AcadosModel object
        model = AcadosModel()
        model.f_expl_expr = state_derivatives
        model.x = states
        model.u = controls
        model.name = self.model_name

        return model
