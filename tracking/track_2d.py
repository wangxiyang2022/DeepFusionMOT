# Note：The code code is referenced from https://github.com/nwojke/deep_sort

'''
  2D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated
  with any detections for several frames, it is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_2D:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):

        self.mean = mean
        self.covariance = covariance
        self.track_id_2d = track_id  #
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_2D  # 2D tracking
        self.time_since_update = 0
        self.n_init = n_init    # 连续n_init帧被检测到，状态就被设为confirmed
        self._max_age = max_age  # 一个跟踪对象丢失多少帧后会被删去（删去之后将不再进行特征匹配）

    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`.
        Returns
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def x1y1x2y2(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def  increment(self):
        self.age += 1
        self.time_since_update += 1

    def predict_2d(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment()

    def update_2d(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        # self.features.append(detection.feature)
        self.hits += 1
        # self.age += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state =TrackState.Confirmed


    def mark_missed(self):
        if self.state == TrackState.Tentative or self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
