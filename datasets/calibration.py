# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>measure are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        with open(calib_filepath) as f:
            self.P0 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P1 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Projection matrix from rectified camera coord to image2/3 coord
            self.P2 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P3 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Rotation from reference camera coord to rectified camera coord
            line = f.readline()
            self.R0_rect = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 3))

            # Rigid transform from lidar coord to reference camera coord
            line = f.readline()
            self.Tr_lidar_to_cam = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))  # lidar_to_cam
            self.Tr_cam_to_lidar = inverse_rigid_trans(self.Tr_lidar_to_cam)                       # cam_to_lidar

            line = f.readline()
            self.Tr_imu_to_lidar = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))  # imu_to_lidar
            self.Tr_lidar_to_imu = inverse_rigid_trans(self.Tr_imu_to_lidar)                       # lidar_to_imu

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr