# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from tracking.cost_function import iou_2d, giou_2d, sdiou_2d, diou_2d, giou_3d, dist_3d


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def kitti_cost(dets, trks, iou_threshold, iou_matrix, cost_func):
    matched_indices, _ = cost_calculate(dets, trks, iou_matrix, iou_threshold, cost_func)
    return matched_indices


def cost_calculate(dets, trks, iou_matrix, iou_threshold, cost_func):
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            if cost_func == 'iou_2d':
                iou_matrix[d, t] = iou_2d(det, trk)  # det: 8 x 3, trk: 8 x 3
            elif cost_func == 'giou_2d':
                iou_matrix[d, t] = giou_2d(det, trk)
            elif cost_func == 'sdiou_2d':
                iou_matrix[d, t] = sdiou_2d(det, trk)
            elif cost_func == 'diou_2d':
                iou_matrix[d, t] = diou_2d(det, trk)
            elif cost_func == 'giou_3d' or cost_func == 'iou_3d':
                iou_matrix[d, t] = giou_3d(det, trk, cost_func)
            elif cost_func == 'dist_3d':
                iou_matrix[d, t] = dist_3d(det, trk)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
        # matched_indices = greedy_matching(-iou_matrix)
    return matched_indices, iou_matrix


def associate_dets_to_trks_fusion(dets, trks, cost_func, iou_threshold, metric):
    if (len(trks) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), []
    if (len(dets) == 0):
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks))
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if metric == 'match_3d':
        matched_indices = kitti_cost(dets, trks, iou_threshold, iou_matrix, cost_func)
    # matched_indices = nuscenes_cost(detections, trackers, iou_matrix)
    elif metric == 'match_2d':
        dets = np.array([d.to_x1y1x2y2() for d in dets])
        trks = np.array([t.to_x1y1x2y2() for t in trks])
        matched_indices, _ = cost_calculate(dets, trks, iou_matrix, iou_threshold, cost_func)

    return is_matched(dets, trks, matched_indices, iou_matrix, iou_threshold)


def trackfusion2Dand3D(trks_2d, trks_3Dto2D_image, iou_threshold):
    trk_indices = list(range(len(trks_2d)))  # 跟踪对象索引
    det_indices = list(range(len(trks_3Dto2D_image)))  # 检测对象索引
    matches = []
    if len(trk_indices) == 0 or len(det_indices) == 0:
        return [], trk_indices, det_indices  # Nothing to match.

    iou_matrix = np.zeros((len(trks_2d), len(trks_3Dto2D_image)), dtype=np.float32)
    for t, trk in enumerate(trks_2d):
        for d, det in enumerate(trks_3Dto2D_image):
            iou_matrix[t, d] = iou_2d(trk.to_x1y1x2y2(), det)  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_dets = []
    for d, det in enumerate(trks_3Dto2D_image):
        if d not in matched_indices[:, 1]:
            unmatched_dets.append(d)

    unmatched_trks_2d = []
    for t, trk in enumerate(trks_2d):
        if t not in matched_indices[:, 0]:
            unmatched_trks_2d.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[1])
            unmatched_trks_2d.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trks_2d), np.array(unmatched_dets)


def associate_2D_to_3D_tracking(trks_2d, trks_3d, iou_threshold):
    trks_3Dto2D_image = [list(i.additional_info[2:6]) for i in trks_3d]
    matched_trks_2d, unmatch_trks_2d, _ = trackfusion2Dand3D(trks_2d, trks_3Dto2D_image, iou_threshold)
    return matched_trks_2d, unmatch_trks_2d


def is_matched(dets, trks, matched_indices, iou_matrix, iou_threshold):
    matches, unmatched_dets, unmatched_trks = [], [], []
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    for t, trk in enumerate(trks):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
