import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from scipy.integrate import simpson
import casadi as ca

class Path:
    def __init__(self, path_points):
        self.path_points = path_points
        self.spline_n, self.spline_e, self.spline_phi, self.total_length, self.spline_points = self.get_bspline_curves()

    def ensure_strictly_increasing(self, x):
        """
        Ensure the elements of the array x are strictly increasing.
        """
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                x[i] = x[i - 1] + 1e-9
        return x

    def pad_to_multiple(self, arr, multiple):
        """
        Pad the array to make its length a multiple of the specified number.
        """
        if len(arr) % multiple != 0:
            padding_length = multiple - (len(arr) % multiple)
            arr = np.pad(arr, (0, padding_length), 'constant')
        return arr

    def get_bspline_curves(self):
        """
        Generates B-spline curves for the path and its gradient, and calculates total path length.
        Returns:
            tuple: Spline objects for north coordinates, east coordinates, tangent angles, and total path length.
        """
        waypoints = np.array(self.path_points)
        
        # Create a fine grid for the original waypoints
        u_original = np.linspace(0, 1, len(waypoints))
        
        # Interpolate waypoints to generate additional points
        interp_points = 10 * (len(waypoints) - 1)
        u_fine = np.linspace(0, 1, interp_points)
        spline_n_fine = np.interp(u_fine, u_original, waypoints[:, 0])
        spline_e_fine = np.interp(u_fine, u_original, waypoints[:, 1])
        
        # Fit B-spline to the interpolated points
        tck, _ = splprep([spline_n_fine, spline_e_fine], s=0, k=3)
        tck[0] = self.ensure_strictly_increasing(tck[0])  # Ensure knots are strictly increasing
        tck[1][0] = self.pad_to_multiple(tck[1][0], len(tck[0]))  # Pad to ensure correct number of elements
        tck[1][1] = self.pad_to_multiple(tck[1][1], len(tck[0]))  # Pad to ensure correct number of elements

        # u_fine = u_fine[::-1]
        x_fine, y_fine = splev(u_fine, tck)
        spline_points = np.column_stack((y_fine, x_fine))
        
        # Calculate derivatives
        dx_du, dy_du = splev(u_fine, tck, der=1)
        
        # Calculate tangent angles
        phi_values = np.arctan2(dy_du, dx_du)
        
        # Calculate path length
        ds = np.sqrt(dx_du**2 + dy_du**2)
        total_length = simpson(ds, u_fine)
        
        # Create B-spline representation of the tangent angle
        tck_phi, _ = splprep([u_fine * total_length, phi_values], s=0, k=3)
        tck_phi[0] = self.ensure_strictly_increasing(tck_phi[0])  # Ensure knots are strictly increasing
        tck_phi[1][1] = self.pad_to_multiple(tck_phi[1][1], len(tck_phi[0]))  # Pad to ensure correct number of elements
        
        # Define CasADi functions for evaluation
        self.spline_n_func = ca.interpolant('spline_n', 'bspline', [tck[0]], tck[1][0])
        self.spline_e_func = ca.interpolant('spline_e', 'bspline', [tck[0]], tck[1][1])
        self.spline_phi_func = ca.interpolant('spline_phi', 'bspline', [tck_phi[0]], tck_phi[1][1])
        
        return BSpline(tck[0], tck[1][0], tck[2]), BSpline(tck[0], tck[1][1], tck[2]), BSpline(tck_phi[0], tck_phi[1][1], tck_phi[2]), total_length, spline_points

    def evaluate_path(self, s):
        """
        Evaluate the path at a given distance s along the path.
        Args:
            s (float or MX): Distance along the path, from 0 to total_length.
        Returns:
            tuple: Coordinates (n_ref, e_ref) at the given s.
        """
        theta = s
        
        # Ensure theta is handled properly in CasADi context
        n_ref = self.spline_n_func(theta)
        e_ref = self.spline_e_func(theta)
            
        return n_ref, e_ref

    def get_tangent_angle(self, s):
        """
        Get the tangent angle at a given distance s along the path.
        Args:
            s (float or MX): Distance along the path, from 0 to total_length.
        Returns:
            float: Tangent angle (phi) at the given s.
        """
        theta = s
        
        # Ensure theta is handled properly in CasADi context
        phi_ref = self.spline_phi_func(theta)
        
        return phi_ref
    
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
