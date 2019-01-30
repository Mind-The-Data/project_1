import numpy as np
from math import sin, cos
import scipy.optimize as so
import matplotlib.pyplot as plt


class Camera(object):
    def __init__(self, f, c, p):
        self.p = p  # Pose (x_cam, y_cam, z_cam, yaw, pitch, roll)
        self.f = f  # Focal Length in Pixels
        self.c = np.array(c)  # sensor size?

    def transforms(self, X, p):
        """
        This function performs the translation and rotation from world coordinates into generalized camera coordinates.
        X: 3 D array of Easting, Northing, Elev for each point
        """
        ### rotational transform
        # read in real world coordinates
        new_col = np.ones((len(X), 1))
        hom_coords = np.append(X, new_col, 1)

        # R yaw matrix
        R_yaw = np.matrix([[cos(p[3]), -sin(p[3]), 0, 0],
                           [sin(p[3]), cos(p[3]), 0, 0],
                           [0, 0, 1, 0]])

        # R pitch matrix
        R_pitch = np.matrix([[1, 0, 0],
                             [0, cos(p[4]), sin(p[4])],
                             [0, -sin(p[4]), cos(p[4])]])

        # R roll matrix
        R_roll = np.matrix([[cos(p[5]), 0, -sin(p[5])],
                            [0, 1, 0],
                            [sin(p[5]), 0, cos(p[5])]])

        # R axis matrix
        R_axis = np.matrix([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])

        # translation matrix
        T_mat = np.matrix([[1, 0, 0, -p[0]],
                           [0, 1, 0, -p[1]],
                           [0, 0, 1, -p[2]],
                           [0, 0, 0, 1]])

        # C matrix
        C_mat = R_axis @ R_roll @ R_pitch @ R_yaw @ T_mat

        # output generalized coords
        gen_coords = np.zeros((len(X), 3))
        for i in range(len(gen_coords)):
            h_coord_mat = np.matrix(hom_coords[i]).T
            gen_coords[i] = (C_mat@h_coord_mat).T

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

        #print(cam_coords)

        return cam_coords

    def residuals(self, p, X, u_gcp):
        error = self.transforms(X, p).flatten() - u_gcp.flatten()

        return error

    def estimate_pose(self, X, u_gcp):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp
        and the projected pixels coordinates of X_gcp is minimized.
        """

        # Use scipy implementation of Levenburg-Marquardt to find the optimal
        # pose values
        p_opt = so.least_squares(self.residuals, self.p, method='lm', args=(X, u_gcp))['x']  # 'x' is dict key to opt values

        return p_opt

        #plt.plot(x, y_obs, 'k.')
        #plt.plot(x, f(x, p_true), 'r-')
        #plt.plot(x, f(x, p_opt), 'b-')
        #plt.show()

        #pass

    def plot_output(self, im, u_true, v_true, u_est, v_est):
        im = plt.imread(im)
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(im)
        ax.scatter(u_true, v_true, s=60, marker='x', color='red', label='true gcps')
        ax.scatter(u_est, v_est, s=60, marker='x', color='yellow', label='estimated gcps')
        plt.legend()
        plt.show()

        fig.savefig('gcps.png', dpi=150)

        pass


f = 3200.  # focal length
c = [4608., 3456.]  # sensor size
p = [272500, 5193700, 1000, 0.78, 0.2, 0]
coords_txt = 'coords.csv'
image = 'Clapp.png'

coords = np.loadtxt(coords_txt, delimiter=',', skiprows=1)

X = coords[:, 2:]
u_gcp = coords[:, 0:2]

cam = Camera(f, c, p)
out = cam.estimate_pose(X, u_gcp)
print(out)

np.savetxt('optimum_pose.txt', out, delimiter=' ')

cam_coords = cam.transforms(X, out)
cam.plot_output(image, u_gcp[:,0], u_gcp[:,1], cam_coords[:,0], cam_coords[:,1])