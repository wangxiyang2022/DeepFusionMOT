# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np


class Detection_2D(object):
    def __init__(self, tlwh):
        '''
        :param tlwh:  top_left x   top_left y    width   height
        :param additional_info:
        '''
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        # self.feature = np.asarray(feature, dtype=np.float32)

    def to_x1y1x2y2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class Detection_3D_Fusion(object):
    def __init__(self, BBox_3D, additional_info):
        self.bbox = np.asarray(BBox_3D, dtype=np.float)
        self.additional_info = np.asarray(additional_info, dtype=np.float32)

class Detection_3D_only(object):
    def __init__(self, BBox_3D, additional_info):
        self.bbox = np.asarray(BBox_3D, dtype=np.float)
        self.additional_info = np.asarray(additional_info, dtype=np.float32)

