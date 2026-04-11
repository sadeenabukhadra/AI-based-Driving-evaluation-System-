import numpy as np
from collections import defaultdict
from parking_zone import ParkingZone
from config import CALIB_FRAMES, CALIB_VOTE_THRESH, REAL_CONE_DIST_CM


class Calibrator:
    def __init__(self):
        self.frame_count = 0
        self.cone_votes  = defaultdict(list)
        self.done        = False
        self.zone        = None

    @property
    def progress(self):
        return min(self.frame_count / max(CALIB_FRAMES, 1), 1.0)

    def update(self, cone_centers):
        if self.done:
            return

        self.frame_count += 1

        for cx, cy in cone_centers:
            matched = False
            for key, pts in self.cone_votes.items():
                mx, my = np.mean(pts, axis=0)
                if abs(cx - mx) < 40 and abs(cy - my) < 40:
                    pts.append((cx, cy))
                    matched = True
                    break
            if not matched:
                new_key = len(self.cone_votes)
                self.cone_votes[new_key].append((cx, cy))

        if self.frame_count >= CALIB_FRAMES:
            self._finalize()

    def _finalize(self):
        stable = []
        for pts in self.cone_votes.values():
            if len(pts) >= CALIB_FRAMES * CALIB_VOTE_THRESH:
                mean = np.mean(pts, axis=0)
                stable.append(tuple(mean.astype(int)))

        if len(stable) >= 2:
            self.zone = ParkingZone(stable, REAL_CONE_DIST_CM)
            self.done = True
            print(f"Calibration done — {len(stable)} stable cone(s) found.")
        else:
            print(f"Only {len(stable)} stable cone(s) found — retrying.")
            self.frame_count = 0
            self.cone_votes  = defaultdict(list)