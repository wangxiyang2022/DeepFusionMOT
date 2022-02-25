# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers_fusion(detections, trackers, iou_threshold):
	"""
	Assigns detections to tracked object (both represented as bounding boxes)
    detections:  N x 8 x 3
	trackers:    M x 8 x 3
	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
	if len(dets_8corner) > 0:
		dets_8corner = np.stack(dets_8corner, axis=0)
	else:
		dets_8corner = []

	trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in trackers]
	if len(trks_8corner) > 0:
		trks_8corner = np.stack(trks_8corner, axis=0)
	if (len(trks_8corner)==0):
		return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

	iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
	# eucliDistance_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
	for d, det in enumerate(dets_8corner):
		for t, trk in enumerate(trks_8corner):
			iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3

	matches = []
	if min(iou_matrix.shape) > 0:
		a = (iou_matrix > iou_threshold).astype(np.int32)
		if a.sum(1).max() == 1 and a.sum(0).max() == 1:
			matched_indices = np.stack(np.where(a),axis=1)
		else:
			matched_indices = linear_assignment(-iou_matrix)
	else:
		matched_indices = np.empty(shape=(0, 2))

	unmatched_detections = []
	for d, det in enumerate(dets_8corner):
		if d not in matched_indices[:, 0]:
			unmatched_detections.append(d)

	unmatched_trackers = []
	for t, trk in enumerate(trks_8corner):
		if t not in matched_indices[:, 1]:
			unmatched_trackers.append(t)

	for m in matched_indices:
		if iou_matrix[m[0], m[1]] < iou_threshold:
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1, 2))
	if len(matches) == 0:
		matches = np.empty((0, 2), dtype=int)
	else:
		matches = np.concatenate(matches, axis=0)
	# else:
	# 	matches = []
	# 	# calculate Euclidean distance
	# 	for d, det in enumerate(detections):
	# 		for t, trk in enumerate(trackers):
	# 			eucliDistance_matrix[d, t] = eucliDistance(det.bbox[0:3], trk.pose[0:3])
	# 	# eucliDistance_matrix = np.where(eucliDistance_matrix < 1, eucliDistance_matrix, 0)
	#
	# 	if not np.all(eucliDistance_matrix == 0):
	# 		row_ind, col_ind = linear_sum_assignment(eucliDistance_matrix)
	# 		matched_indices = np.stack((row_ind, col_ind), axis=1)
	#
	# 		unmatched_detections = []
	# 		for d, det in enumerate(dets_8corner):
	# 			if d not in matched_indices[:, 0]:
	# 				unmatched_detections.append(d)
	#
	# 		unmatched_trackers = []
	# 		for t, trk in enumerate(trks_8corner):
	# 			if t not in matched_indices[:, 1]:
	# 				unmatched_trackers.append(t)
	#
	# 		for m in matched_indices:
	# 			if eucliDistance_matrix[m[0], m[1]] >= 1.5:
	# 				unmatched_detections.append(m[0])
	# 				unmatched_trackers.append(m[1])
	# 				pass
	# 			else:
	# 				matches.append(m.reshape(1, 2))
	# 		if len(matches) == 0:
	# 			matches = np.empty((0, 2), dtype=int)
	# 		else:
	# 			matches = np.concatenate(matches, axis=0)
	# 	else:
	# 		matches, unmatched_detections, unmatched_trackers = [], [], []

	# print('----')

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def trackfusion2Dand3D(tracker_2D, trks_3Dto2D_image, iou_threshold):
    track_indices = list(range(len(tracker_2D)))  # 跟踪对象索引
    detection_indices = list(range(len(trks_3Dto2D_image)))  # 检测对象索引
    matches = []
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    iou_matrix = np.zeros((len(tracker_2D), len(trks_3Dto2D_image)), dtype=np.float32)
    for t, trk in enumerate(tracker_2D):
        for d, det in enumerate(trks_3Dto2D_image):
            iou_matrix[t, d] = iou_batch(trk.x1y1x2y2(), det)  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(trks_3Dto2D_image):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_tracker_2D = []
    for t, trk in enumerate(tracker_2D):
        if t not in matched_indices[:, 0]:
            unmatched_tracker_2D.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracker_2D.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_tracker_2D), np.array(unmatched_detections)

def associate_2D_to_3D_tracking(tracker_2D, tracks_3D, calib_file, iou_threshold):
	trks_3Dto2D_image = [list(i.additional_info[2:6])  for i in tracks_3D]
	matched_track_2D, unmatch_tracker_2D, _ = trackfusion2Dand3D(tracker_2D, trks_3Dto2D_image, iou_threshold)
	return matched_track_2D, unmatch_tracker_2D