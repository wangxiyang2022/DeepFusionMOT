# Author: wangxy
# Emial: 1393196999@qq.com
import numpy as np
from tracking.cost_function import iou_batch, convert_3dbox_to_8corner
from tracking.matching import linear_assignment


def datafusion2Dand3D(detections_3D_camera, detection_2D, detection_3Dto2D, additional_info):
    '''
    :param detections_3D_camera:  Detections by LiDAR
    :param detection_2D:          Detections by Camera
    :param detection_3Dto2D:      Detections by LiDAR
    :return:
        detection_2D_fusion: Objects detected by both LiDAR and camera at the same time(using 2D BBox to representation in the pixel domain).
        detection_2D_only: Objects detected by camera only.
        detection_3D_fusion:  Objects detected by both LiDAR and camera at the same time(using 3D BBox to representation in the camera domain).
        detection_3D_only: Objects detected by LiDAR only.
    '''
    iou_threshold = 0.3
    iou_matrix = np.zeros((len(detection_2D), len(detection_3Dto2D)), dtype=np.float32)
    for d1, det_2D in enumerate(detection_2D):
        for d2, det_3Dto2D in enumerate(additional_info):
            iou_matrix[d1, d2] = iou_batch(det_2D, det_3Dto2D[2:6])  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a),axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    matched, unmatched_detections_2D, unmatched_detection_3Dto2D = [], [], []
    detection_3D_fusion_info, detection_3D_only_info = [],[]
    for d, det in enumerate(detection_2D):
        if d not in matched_indices[:, 0]:
            unmatched_detections_2D.append(d)

    for t, trk in enumerate(detection_3Dto2D):
        if t not in matched_indices[:, 1]:
            unmatched_detection_3Dto2D.append(t)
            # unmatched_3Dto2D_additional_info.append(additional_info[t])

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections_2D.append(m[0])
            unmatched_detection_3Dto2D.append(m[1])
        else:
            matched.append(m.reshape(1, 2))

    if len(matched) == 0:
        matched = np.empty((0, 2), dtype=int)
    else:
        matched = np.concatenate(matched, axis=0)

    detection_2D_fusion, detection_3Dto2D_fusion, detection_3D_fusion, detection_2D_only, \
    detection_2Dto3D_only, detection_3D_only = [],[],[],[],[],[]

    for detection_2D_idx, detection_3Dto2D_idx in matched:
        detection_2D_fusion.append(detection_2D[detection_2D_idx].tolist())
        detection_3D_fusion_info.append(additional_info[detection_3Dto2D_idx])
        detection_3Dto2D_fusion.append(detection_3Dto2D[detection_3Dto2D_idx].tolist())
        detection_3D_fusion.append(detections_3D_camera[detection_3Dto2D_idx].tolist())

    for unmatched_detections_2D_idx in unmatched_detections_2D:
        detection_2D_only.append(detection_2D[unmatched_detections_2D_idx].tolist())

    for unmatched_detections_2Dto3D_idx in unmatched_detection_3Dto2D:
        detection_2Dto3D_only.append(detection_3Dto2D[unmatched_detections_2Dto3D_idx].tolist())
        detection_3D_only.append(detections_3D_camera[unmatched_detections_2Dto3D_idx].tolist())
        detection_3D_only_info.append(additional_info[unmatched_detections_2Dto3D_idx])

    detection_3D_fusion = {'dets_3d_fusion': detection_3D_fusion, 'dets_3d_fusion_info': detection_3D_fusion_info}
    detection_3D_only = {'dets_3d_only': detection_3D_only, 'dets_3d_only_info': detection_3D_only_info}

    return np.array(detection_2D_fusion), np.array(detection_3Dto2D_fusion), detection_3D_fusion, \
               np.array(detection_2D_only), np.array(detection_2Dto3D_only), detection_3D_only
