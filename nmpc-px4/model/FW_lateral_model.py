from acados_template import AcadosModel
import casadi as cs
import numpy as np

class FixedWingLateralModel:
    def __init__(self, model_name='fixed_wing_lateral'):
        self.model_name = model_name
        self.gravity = 9.81

    def fixed_wing_lateral_model(self) -> AcadosModel:
        # State variables (assuming no wind)
        I_n = cs.MX.sym('I_n')      # north position
        I_e = cs.MX.sym('I_e')      # east position
        B_v_x = cs.MX.sym('B_v_x')  # velocity_x
        B_v_y = cs.MX.sym('B_v_y')  # velocity_y
        I_yaw = cs.MX.sym('I_yaw')  # yaw angle
        states = cs.vertcat(I_n, I_e, B_v_x, B_v_y, I_yaw)

        # Input variables
        B_a_x = cs.MX.sym('B_a_x')      # acceleration x
        B_a_y = cs.MX.sym('B_a_y')      # acceleration y
        I_yaw_rate = cs.MX.sym('I_yaw_rate')  # yaw rate
        controls = cs.vertcat(B_a_x, B_a_y, I_yaw_rate)


        # Parameteres
        p = cs.MX.sym('p', 8)
        w_n = p[0]
        w_e = p[1]

        V_air_n = B_v_x * cs.cos(I_yaw) + w_n
        V_air_e = B_v_x * cs.sin(I_yaw) + w_e
        a_v = cs.sqrt(V_air_n**2 + V_air_e**2)

        # Define the dynamics equations
        dn_dt = B_v_x * cs.cos(I_yaw) - B_v_y*cs.sin(I_yaw) + w_n # derivative of north position
        de_dt = B_v_x * cs.sin(I_yaw) + B_v_y*cs.cos(I_yaw) + w_e # derivative of east position
        dv_x_dt = B_a_x - B_v_y*I_yaw_rate #(self.gravity * cs.tan(I_roll) / a_v) # derivative of velocity in x
        dv_y_dt = B_a_y + B_v_x*I_yaw_rate #(self.gravity * cs.tan(I_roll) / a_v) # derivative of velocity in y
        dyaw_dt = I_yaw_rate #self.gravity * cs.tan(I_roll) / a_v  # derivative of yaw angle

        # Concatenate the state derivatives
        state_derivatives = cs.vertcat(dn_dt, de_dt, dv_x_dt, dv_y_dt, dyaw_dt)

        # Create AcadosModel object
        model = AcadosModel()
        model.f_expl_expr = state_derivatives  # explicit ODE right-hand side
        model.x = states  # state vector
        model.u = controls  # control vector
        model.name = self.model_name  # model name
        model.p = p  # parameters

        return model
    
    @staticmethod
    def np_wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def cs_wrap_angle(angle):
        return cs.fmod(angle + cs.pi, 2*cs.pi) - cs.pi

