import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from scipy.integrate import cumtrapz
from scipy.optimize import minimize_scalar
import casadi as ca
import matplotlib.pyplot as plt
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

        # Unwrap the angles to remove 2π jumps
        phi_values_unwrapped = np.unwrap(phi_values)

        # Calculate path length
        ds = np.sqrt(dx_du**2 + dy_du**2)
        s = cumtrapz(ds, u_fine, initial=0)
        total_length = s[-1]

        # Normalize s to [0, 1]
        s_normalized = s / total_length

        # Fit B-spline to unwrapped phi values
        tck_phi, _ = splprep([s_normalized, phi_values_unwrapped], s=0, k=3)
        
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
    
    def project_to_path(self, x, y, yaw, dt, velocity_x, velocity_y, params, initial=True):
        x, y = float(x), float(y)

        def distance(t):
            path_x, path_y = self.evaluate_path(t * self.total_length)
            return (path_x - x)**2 + (path_y - y)**2

        result = minimize_scalar(distance, bounds=(0, 1), method='bounded', options={'disp': False})
        
        theta = result.x * self.total_length

        return theta
    
    def plot_tangent_angles(self):
        thetas = np.linspace(0, self.total_length, 1000)
        angles = [self.get_tangent_angle(t) for t in thetas]
        
        plt.figure(figsize=(12, 6))
        plt.plot(thetas, angles)
        plt.title('Tangent Angles Along Path')
        plt.xlabel('Path Length')
        plt.ylabel('Angle (radians)')
        plt.grid(True)
        plt.show()

        # Plot rate of change
        angle_rates = np.diff(angles) / np.diff(thetas)
        plt.figure(figsize=(12, 6))
        plt.plot(thetas[1:], angle_rates)
        plt.title('Rate of Change of Tangent Angles')
        plt.xlabel('Path Length')
        plt.ylabel('Rate of Change (radians/meter)')
        plt.grid(True)
        plt.show()
    
