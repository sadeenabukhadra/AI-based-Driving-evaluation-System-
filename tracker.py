import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.bbox = bbox
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.x[:4] = np.array(bbox).reshape((4,1))
        self.time_since_update = 0

    def update(self, bbox):
        self.kf.update(np.array(bbox))
        self.bbox = bbox
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.bbox


class SORT:
    def __init__(self):
        self.tracks = []
        self.track_id = 0

    def update(self, detections):
        objects = []

        for det in detections:
            bbox = det["box"]

            matched = False
            for track in self.tracks:
                track.update(bbox)
                objects.append({
                    "id": track.id,
                    "box": bbox,
                    "class": det["class"]
                })
                matched = True
                break

            if not matched:
                t = Track(bbox, self.track_id)
                self.track_id += 1
                self.tracks.append(t)

                objects.append({
                    "id": t.id,
                    "box": bbox,
                    "class": det["class"]
                })

        return objects