import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from scipy.integrate import cumtrapz
from scipy.optimize import minimize_scalar
import casadi as ca

class Path:
    def __init__(self, path_points, num_laps):
        self.path_points = path_points
        self.num_laps = num_laps
        self.spline_x, self.spline_y, self.spline_phi, self.total_length, self.spline_points = self.get_bspline_curves()
        self.is_closed = self.check_if_closed()
        self.extended_length = self.total_length * self.num_laps if self.is_closed else self.total_length

    def check_if_closed(self):
        start_point = np.array(self.path_points[0])
        end_point = np.array(self.path_points[-1])
        return np.linalg.norm(start_point - end_point) < 1e-3

    def get_bspline_curves(self):
        waypoints = np.array(self.path_points)
        
        # Fit B-spline to the waypoints (x, y)
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=3)  # Swapped 0 and 1
        
        # Generate a fine representation of the spline
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)
        spline_points = np.column_stack((x_fine, y_fine))
        
        # Calculate derivatives
        dx_du, dy_du = splev(u_fine, tck, der=1)
        
        # Calculate tangent angles
        phi_values = np.arctan2(dy_du, dx_du)
        
        # Calculate path length
        ds = np.sqrt(dx_du**2 + dy_du**2)
        s = cumtrapz(ds, u_fine, initial=0)
        total_length = s[-1]
        
        # Normalize s to [0, 1]
        s_normalized = s / total_length
        
        # Fit B-spline to phi values
        tck_phi, _ = splprep([s_normalized, phi_values], s=0, k=3)
        
        # Create CasADi functions for evaluation
        self.spline_x_func = ca.interpolant('spline_x', 'bspline', [u], tck[1][0])
        self.spline_y_func = ca.interpolant('spline_y', 'bspline', [u], tck[1][1])
        self.spline_phi_func = ca.interpolant('spline_phi', 'bspline', [s_normalized], tck_phi[1][1])
        
        return (BSpline(tck[0], tck[1][0], tck[2]), 
                BSpline(tck[0], tck[1][1], tck[2]), 
                BSpline(tck_phi[0], tck_phi[1][1], tck_phi[2]), 
                total_length,
                spline_points)

    def evaluate_path(self, theta):
        if self.is_closed:
            theta = ca.fmod(theta, self.total_length)
        
        if isinstance(theta, (float, int)):
            x_ref = float(self.spline_x_func(theta/self.total_length))
            y_ref = float(self.spline_y_func(theta/self.total_length))
        else:
            x_ref = self.spline_x_func(theta/self.total_length)
            y_ref = self.spline_y_func(theta/self.total_length)
        return x_ref, y_ref

    def get_tangent_angle(self, theta):
        if self.is_closed:
            theta = ca.fmod(theta, self.total_length)
        
        if isinstance(theta, (float, int)):
            return float(self.spline_phi_func(theta/self.total_length))
        else:
            return self.spline_phi_func(theta/self.total_length)
    
    def project_to_path(self, x, y):
        x, y = float(x), float(y)

        def distance(t):
            path_x, path_y = self.evaluate_path(t * self.total_length)
            return (path_x - x)**2 + (path_y - y)**2

        result = minimize_scalar(distance, bounds=(0, 1), method='bounded', options={'disp': False})
        
        theta = result.x * self.total_length
        return theta