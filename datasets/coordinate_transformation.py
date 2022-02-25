import copy
import numpy as np
from datasets.calibration import Calibration


class TransformationKitti(Calibration):
    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def __init__(self, calib_file):
        super().__init__(calib_file)

    def project_lidar_to_ref(self, pts_3d_lidar):
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)  # nx4
        return np.dot(pts_3d_lidar, np.transpose(self.Tr_lidar_to_cam))

    def project_imu_to_lidar(self, pts_3d_imu):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in IMU coord.
        '''
        pts_3d_imu = self.cart2hom(pts_3d_imu)  # nx4
        return np.dot(pts_3d_imu, np.transpose(self.Tr_imu_to_lidar))


    def project_lidar_to_imu(self, pts_3d_lidar):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in IMU coord.
        '''
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)  # nx4
        return np.dot(pts_3d_lidar, np.transpose(self.self.Tr_lidar_to_imu))


    def project_ref_to_lidar(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.self.Tr_cam_to_lidar))


    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0_rect), np.transpose(pts_3d_rect)))


    def project_ref_to_rect(self, pts_3d_ref):
        '''
        Input and Output are nx3 points
        '''
        # pts_3d_ref = Tr_lidar_to_cam * [x y z 1]
        return np.transpose(np.dot(self.R0_rect, np.transpose(pts_3d_ref)))  # R0_rect_rect * Tr_lidar_to_cam * A

    def project_rect_to_lidaro(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in lidar coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_lidar(pts_3d_ref)


    def project_lidar_to_rect(self, pts_3d_lidar):
        pts_3d_ref = self.project_lidar_to_ref(pts_3d_lidar)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================

    def project_rect_to_image(self, pts_3d_rect):
        '''
            Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(
            self.P2))  # nx3     P_rect_2 * R0_rect_rect *Tr_lidar_to_cam * A
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        x1y1, x2y2 = np.min(pts_2d[:, 0:2], axis=0).tolist(), np.max(pts_2d[:, 0:2], axis=0).tolist()
        pts_2d_img = x1y1 + x2y2
        for idx, value in enumerate(pts_2d_img):
            if value <= 0:
                pts_2d_img[idx] = 0
            if value >= 1241:
                pts_2d_img[idx] = 1241
        return pts_2d_img

    def project_3d_to_image(self, pts_3d_rect):
        '''
            Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))  # nx3     P_rect_2 * R0_rect_rect *Tr_lidar_to_cam * A
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:,0:2]

    def project_lidar_to_image(self, pts_3d_lidar):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in image2 coord.
        '''
        pts_3d_rect = self.project_lidar_to_rect(pts_3d_lidar)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_lidar(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_lidar(pts_3d_rect)


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object's 3D box with the representation of [h,w,l, x,y,z,theta] and
        convert it to the 8 corners of the 3D box

        Returns:
            corners_3d: (8,3) array in in rect camera coord
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[6])

    # 3d bounding box dimensions
    l = bbox3d[2]
    w = bbox3d[1]
    h = bbox3d[0]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners, y_corners, z_corners]))  # np.vstack([x_corners,y_corners,z_corners])
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[3]  # x
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[4]  # y
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[5]  # z

    a = np.transpose(corners_3d)
    return np.transpose(corners_3d)

def compute_box_3dto2d(bbox3d_input, calib_file):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[6])

    # 3d bounding box dimensions
    l = bbox3d[2]
    w = bbox3d[1]
    h = bbox3d[0]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[3]  # x
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[4]  # y
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[5]  # z
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d
    corners_3d = np.transpose(corners_3d)
    corners_2d = TransformationKitti(calib_file).project_3d_to_image(corners_3d)
    return corners_2d


def convert_x1y1x2y2_to_xywh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, w, h]).tolist()


def convert_x1y1x2y2_to_tlwh(bbox):
    '''
    :param bbox: x1 y1 x2 y2
    :return: tlwh: top_left x   top_left y    width   height
    '''
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array(([bbox[0], bbox[1], w, h]))