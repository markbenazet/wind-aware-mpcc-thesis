from Paths.waypoints import path_points
from scipy.interpolate import make_interp_spline
import numpy as np
import casadi as cs

class Path:
    def __init__(self):
        self.path_points = path_points
        self.curve = self.get_bspline_curve()

    def get_bspline_curve(self, n_points=2000):
        waypoints = np.array(self.path_points, dtype=float)
        n = len(waypoints)
        t = np.linspace(0, 1, n)
        t_fine = np.linspace(0, 1, n_points)
        
        spline_x = make_interp_spline(t, waypoints[:, 0], k=3)
        spline_y = make_interp_spline(t, waypoints[:, 1], k=3)
        
        curve_E = spline_x(t_fine)
        curve_N = spline_y(t_fine)
        
        curve = np.vstack((curve_N, curve_E)).T
        return curve.astype(float)

    def evaluate_path(self, theta):
        theta_fine = np.linspace(0, 1, len(self.curve))
        N = np.interp(theta, theta_fine, self.curve[:, 0])
        E = np.interp(theta, theta_fine, self.curve[:, 1])
        return N, E

    def get_tangent_angle(self, theta):
        theta_fine = np.linspace(0, 1, len(self.curve))
        spline_x = make_interp_spline(theta_fine, self.curve[:, 0], k=3)
        spline_y = make_interp_spline(theta_fine, self.curve[:, 1], k=3)
        
        dx_dtheta = spline_x(theta, 1)
        dy_dtheta = spline_y(theta, 1)
        
        phi = np.arctan2(dy_dtheta, dx_dtheta)
        return phi

    def evaluate_path_symbolic(self, theta):
        theta_fine = np.linspace(0, 1, len(self.curve))
        N = cs.interpolant('N', 'linear', [theta_fine], self.curve[:, 0])(theta)
        E = cs.interpolant('E', 'linear', [theta_fine], self.curve[:, 1])(theta)
        return N, E

    def get_tangent_angle_symbolic(self, theta):
        theta_fine = np.linspace(0, 1, len(self.curve))
        spline_x = cs.interpolant('dx', 'linear', [theta_fine], np.gradient(self.curve[:, 0], theta_fine))
        spline_y = cs.interpolant('dy', 'linear', [theta_fine], np.gradient(self.curve[:, 1], theta_fine))
        
        dx_dtheta = spline_x(theta)
        dy_dtheta = spline_y(theta)
        
        phi = cs.atan2(dy_dtheta, dx_dtheta)
        return phi
