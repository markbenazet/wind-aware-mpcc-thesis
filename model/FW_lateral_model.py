from acados_template import AcadosModel
import casadi as cs
import numpy as np

class FixedWingLateralModel:
    def __init__(self, model_name='fixed_wing_lateral'):
        self.model_name = model_name

    def fixed_wing_lateral_model(self) -> AcadosModel:
        # State variables (assuming no wind)
        I_x = cs.MX.sym('I_x')      # north position
        I_y = cs.MX.sym('I_y')      # east position
        B_v_x = cs.MX.sym('B_v_x')  # velocity_x
        B_v_y = cs.MX.sym('B_v_y')  # velocity_y
        I_yaw = cs.MX.sym('I_yaw')  # yaw angle
        Theta = cs.MX.sym('Theta')
        ay = cs.MX.sym('ay')        # actual y acceleration
        ay_dot = cs.MX.sym('ay_dot')  # rate of change of y acceleration
        states = cs.vertcat(I_x, I_y, B_v_x, B_v_y, I_yaw, Theta, ay, ay_dot)

        # Input variables
        B_a_x = cs.MX.sym('B_a_x')      # acceleration x
        B_a_y = cs.MX.sym('B_a_y')      # acceleration y
        I_yaw_rate = cs.MX.sym('I_yaw_rate')  # yaw rate
        v_k = cs.MX.sym('speed')
        controls = cs.vertcat(B_a_x, B_a_y, I_yaw_rate, v_k)


        # Parameters
        p = cs.MX.sym('p', 4)
        w = p[0:2]
        w_x, w_y = w[0], w[1]
        wn, zeta = p[2], p[3]  # natural frequency and damping ratio for ay dynamics

        # Define the dynamics equations
        dy_dt = B_v_x * cs.cos(I_yaw) - B_v_y*cs.sin(I_yaw) + w_y
        dx_dt = B_v_x * cs.sin(I_yaw) + B_v_y*cs.cos(I_yaw) + w_x
        dv_x_dt = B_a_x + B_v_y*I_yaw_rate
        dv_y_dt = ay - B_v_x*I_yaw_rate
        dyaw_dt = I_yaw_rate
        dtheta_dt = v_k
        day_dt = ay_dot
        day_dot_dt = wn**2 * (B_a_y - ay) - 2*zeta*wn*ay_dot
        

        # Concatenate the state derivatives
        state_derivatives = cs.vertcat(dx_dt, dy_dt, dv_x_dt, dv_y_dt, dyaw_dt, dtheta_dt, day_dt, day_dot_dt)

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

