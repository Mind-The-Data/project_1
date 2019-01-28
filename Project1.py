import numpy as np
from math import sin, cos
import scipy.optimize as so
import matplotlib.pyplot as plt


class Camera(object):
    def __init__(self, gcp_coords):
        self.p = None  # Pose (x_cam, y_cam, z_cam, yaw, pitch, roll)
        self.f = None  # Focal Length in Pixels
        self.c = np.array([None, None])  # sensor size?
        self.gcp_coords = gcp_coords

    def transforms(self, X):
        """
        This function performs the translation and rotation from world coordinates into generalized camera coordinates.
        """
        ### rotational transform
        # read in real world coordinates
        gcp_coords = np.loadtxt(X)
        new_col = np.ones((len(gcp_coords), 1))
        hom_coords = np.append(gcp_coords, new_col, 1)

        # R yaw matrix
        R_yaw = np.matrix([[cos(self.p[3]), -sin(self.p[3]), 0, 0],
                           [sin(self.p[3]), cos(self.p[3]), 0, 0],
                           [0, 0, 1, 0]])

        # R pitch matrix
        R_pitch = np.matrix([[1, 0, 0],
                             [0, cos(self.p[4]), sin(self.p[4])],
                             [0, -sin(self.p[4]), cos(self.p[4])]])

        # R roll matrix
        R_roll = np.matrix([[cos(self.p[5]), 0, -sin(self.p[5])],
                            [0, 1, 0],
                            [sin(self.p[5]), 0, cos(self.p[5])]])

        # R axis matrix
        R_axis = np.matrix([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])

        # translation matrix
        T_mat = np.matrix([[1, 0, 0, -self.p[0]],
                           [0, 1, 0, -self.p[1]],
                           [0, 0, 1, -self.p[2]],
                           [0, 0, 0, 1]])

        # C matrix
        C_mat = R_axis @ R_roll @ R_pitch @ R_yaw @ T_mat

        # output generalized coords
        gen_coords = np.zeros((len(self.gcp_coords), 3))
        for i in range(len(gen_coords)):
            h_coord_mat = np.matrix(hom_coords[i])
            gen_coords[i] = C_mat@h_coord_mat

        ### projective transformation
        p_0 = gen_coords[:, 0]
        p_1 = gen_coords[:, 1]
        p_2 = gen_coords[:, 2]

        x_gen = p_0 / p_2
        y_gen = p_1 / p_2
        c_x = self.c[1] / 2
        c_y = self.c[0] / 2

        u = (self.f * x_gen) + c_x
        v = (self.f * y_gen) + c_y

        cam_coords = np.zeros((len(gen_coords), 2))
        for x in range(len(cam_coords)):
            cam_coords[x][0] = u[x]
            cam_coords[x][1] = v[x]

        return cam_coords

    def estimate_pose(self, X, u_gcp):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp
        and the projected pixels coordinates of X_gcp is minimized.
        """

        # Define our model
        def f(X):
            cam_coords = self.transforms(X)
            return cam_coords

        # Define the residual as f(intercept,slope,x) - y
        def residual(X, u_gcp): # pass u_gcp as 2D array

            return f(X).flatten() - u_gcp.flatten()

        # Make an initial guess about
        p_0 = np.array([0, 0, 0, 0, 0, 0])

        # Use scipy implementation of Levenburg-Marquardt to find the optimal
        # slope and intercept values.
        p_opt = so.least_squares(residual, p_0, method='lm', args=(X, u_gcp))['x']
        self.p = p_opt

        #plt.plot(x, y_obs, 'k.')
        #plt.plot(x, f(x, p_true), 'r-')
        #plt.plot(x, f(x, p_opt), 'b-')
        #plt.show()

        #pass