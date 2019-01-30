# imports
import numpy as np
from math import sin, cos
import scipy.optimize as so
import matplotlib.pyplot as plt


class Camera(object):
    """
    This class contains methods to perform rotational and projective transforms,
    and to estimate camera pose from GCPs
    """
    def __init__(self, f, c, p):
        """

        :param f: camera focal length
        :param c: camera sensor size - [width, height]
        :param p: intial pose estimate - [easting, northing, elevation, yaw, pitch, roll]
        """
        self.p = p  # Pose (x_cam, y_cam, z_cam, yaw, pitch, roll)
        self.f = f  # Focal Length in Pixels
        self.c = np.array(c)  # sensor size [u, v]

    def transforms(self, X, p):
        """
        This method takes real world coordinates, performs a roational transformation to shift them into generalized
        camera coordinates, and then a projective transform to get them into camera sensor coordinates.
        :param X: coordinates [Easting, Northing, Elev]
        :param p: pose
        :return: camera coordinates - [u, v]
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

        return cam_coords

    def residuals(self, p, X, u_gcp):
        """
        Find the residuals between estimated gcp location based on pose, and real gcp location
        :param p: pose
        :param X: real world coordinates
        :param u_gcp: camera coordinates of gcps
        :return: error estimate
        """
        error = self.transforms(X, p).flatten() - u_gcp.flatten()

        return error

    def estimate_pose(self, X, u_gcp):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp
        and the projected pixels coordinates of X_gcp is minimized.
        :param X: real world coordinates
        :param u_gcp: camera coordinates of gcps - [u, v]
        :return: optimum pose estimate
        """

        # Use scipy implementation of Levenburg-Marquardt to find the optimal
        # pose values
        p_opt = so.least_squares(self.residuals, self.p, method='lm', args=(X, u_gcp))['x']  # 'x' is dict key to opt values

        return p_opt

    def plot_output(self, im, u_true, v_true, u_est, v_est):
        """
        plots the location of the gcps in the image along with the estimated gcp locations based on the optimum pose
        :param im: path to image
        :param u_true: measured u coordinates of gcps
        :param v_true: measured v coordinates of gcps
        :param u_est: u coordinates of estimates based on optimum pose
        :param v_est: v coordinates of estimates based on optimum pose
        :return: saves a fig of gcps
        """
        im = plt.imread(im)
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(im)
        ax.scatter(u_true, v_true, s=60, marker='x', color='red', label='true gcps')
        ax.scatter(u_est, v_est, s=60, marker='x', color='yellow', label='estimated gcps')
        plt.legend()
        plt.show()

        fig.savefig('gcps.png', dpi=150)

        pass


# run the camera model for a given image with selected gcps

# params
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