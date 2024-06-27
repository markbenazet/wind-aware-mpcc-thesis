from acados_template import AcadosModel
import casadi as cs

class FixedWingLateralModel:
    def __init__(self, model_name='fixed_wing_lateral'):
        self.model_name = model_name
        self.gravity = 9.81

    def fixed_wing_lateral_model(self) -> AcadosModel:
        # State variables (assuming no wind)
        I_n = cs.SX.sym('I_n')  # north position
        I_e = cs.SX.sym('I_e')  # east position
        I_v = cs.SX.sym('I_v')  # velocity
        I_yaw = cs.SX.sym('I_yaw')  # yaw angle
        states = cs.vertcat(I_n, I_e, I_v, I_yaw)

        # Input variables
        B_a = cs.SX.sym('B_a')  # acceleration
        I_roll = cs.SX.sym('I_roll')  # roll angle
        controls = cs.vertcat(B_a, I_roll)

        # Define the dynamics equations
        dn_dt = I_v * cs.cos(I_yaw)  # derivative of north position
        de_dt = I_v * cs.sin(I_yaw)  # derivative of east position
        dV_dt = B_a  # derivative of velocity
        dyaw_dt = self.gravity * cs.tan(I_roll) / I_v  # derivative of yaw angle

        # Concatenate the state derivatives
        state_derivatives = cs.vertcat(dn_dt, de_dt, dV_dt, dyaw_dt)

        # Create AcadosModel object
        model = AcadosModel()
        model.f_expl_expr = state_derivatives  # explicit ODE right-hand side
        model.x = states  # state vector
        model.u = controls  # control vector
        model.name = self.model_name  # model name

        return model

