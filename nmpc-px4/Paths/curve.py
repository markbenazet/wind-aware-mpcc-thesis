import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from scipy.integrate import cumtrapz
from scipy.optimize import minimize_scalar
import casadi as ca

class Path:
    def __init__(self, path_points):
        self.path_points = path_points
        self.spline_n, self.spline_e, self.spline_phi, self.total_length, self.spline_points = self.get_bspline_curves()

    def get_bspline_curves(self):
        waypoints = np.array(self.path_points)
        
        # Fit B-spline to the waypoints (North, East)
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=3)
        
        # Generate a fine representation of the spline
        u_fine = np.linspace(0, 1, 1000)
        north_fine, east_fine = splev(u_fine, tck)
        spline_points = np.column_stack((north_fine, east_fine))
        
        # Calculate derivatives
        dn_du, de_du = splev(u_fine, tck, der=1)
        
        # Calculate tangent angles (atan2 args order changed to de_du, dn_du)
        phi_values = np.arctan2(dn_du, de_du)
        
        # Calculate path length
        ds = np.sqrt(dn_du**2 + de_du**2)
        s = cumtrapz(ds, u_fine, initial=0)
        total_length = s[-1]
        
        # Normalize s to [0, 1]
        s_normalized = s / total_length
        
        # Fit B-spline to phi values
        tck_phi, _ = splprep([s_normalized, phi_values], s=0, k=3)
        
         # Create CasADi functions for evaluation
        self.spline_n_func = ca.interpolant('spline_n', 'bspline', [u], tck[1][0])
        self.spline_e_func = ca.interpolant('spline_e', 'bspline', [u], tck[1][1])
        self.spline_phi_func = ca.interpolant('spline_phi', 'bspline', [s_normalized], tck_phi[1][1])
        
        return (BSpline(tck[0], tck[1][0], tck[2]), 
                BSpline(tck[0], tck[1][1], tck[2]), 
                BSpline(tck_phi[0], tck_phi[1][1], tck_phi[2]), 
                total_length,
                spline_points)

    def evaluate_path(self, theta):
        if isinstance(theta, (float, int)):
            n_ref = float(self.spline_n_func(theta/self.total_length))
            e_ref = float(self.spline_e_func(theta/self.total_length))
        else:
            n_ref = self.spline_n_func(theta/self.total_length)
            e_ref = self.spline_e_func(theta/self.total_length)
        return n_ref, e_ref

    def get_tangent_angle(self, theta):
        if isinstance(theta, (float, int)):
            return float(self.spline_phi_func(theta/self.total_length))
        else:
            print(self.spline_phi_func(theta/self.total_length))
            return self.spline_phi_func(theta/self.total_length)
    
    def generate_plot_points(self, num_points=1000):
        """
        Generate points along the path for plotting.
        
        Args:
            num_points (int): Number of points to generate.
        
        Returns:
            tuple: Arrays of north and east coordinates.
        """
        s_values = np.linspace(0, self.total_length, num_points)
        north_coords = []
        east_coords = []
        for s in s_values:
            n, e = self.evaluate_path(s)
            north_coords.append(float(n))
            east_coords.append(float(e))
        return np.array(north_coords), np.array(east_coords)
    
    def project_to_path(self, n, e):
        n, e = float(n), float(e)

        def distance(t):
            path_n, path_e = self.evaluate_path(t * self.total_length)
            return (path_n - n)**2 + (path_e - e)**2

        result = minimize_scalar(distance, bounds=(0, 1), method='bounded', options={'disp': False})
        
        theta = result.x * self.total_length
        return theta
