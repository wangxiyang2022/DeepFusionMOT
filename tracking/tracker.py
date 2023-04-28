# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np
from tracking import kalman_filter_2d
from tracking.matching import associate_dets_to_trks_fusion, associate_2D_to_3D_tracking
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D


class Tracker():
    def __init__(self, cfg, category):
        self.cfg = cfg
        self.cost_3d = self.cfg[category].metric_3d
        self.cost_2d = self.cfg[category].metric_2d
        self.threshold_3d = self.cfg[category]["cost_function"][self.cost_3d]
        self.threshold_2d = self.cfg[category]["cost_function"][self.cost_2d]
        self.max_age = self.cfg[category].max_ages
        self.min_frames = self.cfg[category].min_frames
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0   # The id of 3D track is represented by an even number.
        self.track_id_2d = 1   # The id of 3D track is represented by an odd number.
        self.unmatch_tracks_3d = []
        self.kf_2d = kalman_filter_2d.KalmanFilter()

    def predict_3d(self):
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

    def predict_2d(self):
        for track in self.tracks_2d:
            track.predict_2d(self.kf_2d)

    def ego_motion_compensation(self, frame, calib_file, oxts):
        for track in self.tracks_3d:
            track.ego_motion_compensation_3d(frame, calib_file, oxts)

    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_only):
        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d')
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        #  2nd Level of Association
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
            dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d')
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(dets_3d_only[detection_idx])
                    index_to_delete.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        for detection_idx in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # 3rd Level of Association
        matched, unmatch_trks, unmatch_dets = \
            associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_only, self.cost_2d, self.threshold_2d, metric='match_2d')
        for track_idx, detection_idx in matched:
            self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self.initiate_trajectory_2d(dets_2d_only[detection_idx])
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        #  4th Level of Association
        matched_track_2d, unmatch_tracks_2d = associate_2D_to_3D_tracking(self.tracks_2d, self.unmatch_tracks_3d, self.threshold_2d)
        index_to_delete2 = []
        for track_idx_2d, track_idx_3d in matched_track_2d:
            for i in range(len(self.tracks_3d)):
                if self.tracks_3d[i].track_id_3d == self.unmatch_tracks_3d[track_idx_3d].track_id_3d:
                    self.tracks_3d[i].age = self.tracks_2d[track_idx_2d].age + 1
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            index_to_delete2.append(track_idx_2d)
        self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

    def initiate_trajectory_3d(self, detection):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info))
        self.track_id_3d += 2

    def initiate_trajectory_2d(self, detection):
        mean, covariance = self.kf_2d.initiate(detection.to_xyah())
        self.tracks_2d.append(Track_2D(mean, covariance, self.track_id_2d, self.min_frames, self.max_age))
        self.track_id_2d += 2