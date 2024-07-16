from Paths.waypoints import path_points
from scipy.interpolate import make_interp_spline
import numpy as np
import csv
import casadi as ca

class Path:
    def __init__(self, csv_file='path_data.csv'):
        self.path_points = path_points
        self.csv_file = csv_file
        self.curve = self.get_bspline_curve()
        self.total_length = len(self.curve)  # Total number of points in the curve
        self.ref_points, self.theta_values, self.phi_values = self.load_precomputed_data()
        self.n_ref_lut = self.create_interpolant('n_ref')
        self.e_ref_lut = self.create_interpolant('e_ref')
        self.phi_lut = self.create_interpolant('phi')

    def get_bspline_curve(self, n_points=2000):
        """
        Generates a B-spline curve from the given waypoints.
        Args:
            path_points (list): List of waypoints [I_n, I_e].
            n_points (int): Number of points to generate on the B-spline curve.
        Returns:
            numpy.array: Array of points on the B-spline curve.
        """
        waypoints = np.array(self.path_points)
        n = len(waypoints)
        t = np.linspace(0, n - 1, n)  # Use range based on the number of waypoints
        t_fine = np.linspace(0, n - 1, n_points)  # Extend t_fine to the number of points
        
        # Create B-spline representation of the curve
        spline_x = make_interp_spline(t, waypoints[:, 0], k=3)
        spline_y = make_interp_spline(t, waypoints[:, 1], k=3)
        
        # Evaluate the spline on the fine grid
        curve_E = spline_x(t_fine)
        curve_N = spline_y(t_fine)
        
        curve = np.vstack((curve_N, curve_E)).T
        return curve

    def precompute_and_save_data(self):
        """
        Pre-compute reference points, theta values, and phi values,
        and save them to a CSV file.
        """
        theta_fine = np.arange(self.total_length)  # Use integers for theta
        ref_points = self.curve
        phi_values = []

        for theta in theta_fine:
            phi = self.get_tangent_angle(theta)
            phi_values.append(phi)
        
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['theta', 'n_ref', 'e_ref', 'phi'])
            for i, theta in enumerate(theta_fine):
                writer.writerow([theta, ref_points[i, 0], ref_points[i, 1], phi_values[i]])

        print(f"Data saved to {self.csv_file}")

    def load_precomputed_data(self):
        """
        Load pre-computed data from the CSV file.
        Returns:
            numpy.array, numpy.array, numpy.array: Arrays of reference points, theta values, and phi values.
        """
        ref_points = []
        theta_values = []
        phi_values = []
        
        try:
            with open(self.csv_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    theta_values.append(float(row[0]))
                    ref_points.append([float(row[1]), float(row[2])])
                    phi_values.append(float(row[3]))
                    
            ref_points = np.array(ref_points)
            theta_values = np.array(theta_values)
            phi_values = np.array(phi_values)
        except FileNotFoundError:
            print(f"{self.csv_file} not found. Please run precompute_and_save_data() to generate it.")
        
        return ref_points, theta_values, phi_values

    def get_tangent_angle(self, theta):
        """
        Calculate the tangent angle at a given theta.
        Args:
            theta (float): Progress variable from 0 to total length of the curve.
        Returns:
            float: Tangent angle (phi) at the given theta.
        """
        index = int(theta)
        if index + 1 < len(self.curve):
            dx = self.curve[index + 1, 1] - self.curve[index, 1]  # Change in East
            dy = self.curve[index + 1, 0] - self.curve[index, 0]  # Change in North
        else:
            dx = self.curve[index, 1] - self.curve[index - 1, 1]
            dy = self.curve[index, 0] - self.curve[index - 1, 0]
        
        phi = np.arctan2(dx, dy)*180/np.pi  # Note: arctan2(dx, dy) for East-North coordinate system
        return phi
    
    def create_interpolant(self, value_type):
        if value_type == 'n_ref':
            values = self.ref_points[:, 0]
        elif value_type == 'e_ref':
            values = self.ref_points[:, 1]
        elif value_type == 'phi':
            values = self.phi_values
        else:
            raise ValueError("Invalid value_type")
        
        return ca.interpolant(f'{value_type}_lut', 'linear', [self.theta_values], values)

    def evaluate_path(self, theta):
        """
        Evaluate the path at a given theta using pre-computed values and interpolation.
        Args:
            theta (float or MX): Progress variable.
        Returns:
            MX, MX: Coordinates (n_ref, e_ref) at the given theta.
        """
        n_ref = self.n_ref_lut(theta)
        e_ref = self.e_ref_lut(theta)
        return n_ref, e_ref

    def get_tangent_angle_from_precomputed(self, theta):
        """
        Get the tangent angle at a given theta using pre-computed values and interpolation.
        Args:
            theta (float or MX): Progress variable.
        Returns:
            MX: Tangent angle (phi) at the given theta.
        """
        return self.phi_lut(theta)