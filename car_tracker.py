# car_tracker.py
import math
import numpy as np
from config import STABLE_MOVE_PX, STABLE_FRAMES_REQ


class CarState:
    def __init__(self):
        self.center      = None
        self.box         = None
        self.still_count = 0
        self.history     = []
        self.is_stable   = False

    def update(self, new_box):
        x1, y1, x2, y2 = new_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_center = (cx, cy)

        if self.center is not None:
            moved = math.sqrt((cx - self.center[0])**2 +
                              (cy - self.center[1])**2)
            self.still_count = self.still_count + 1 if moved < STABLE_MOVE_PX else 0

        self.center  = new_center
        self.box     = new_box
        self.history.append(new_center)
        if len(self.history) > 20:
            self.history.pop(0)
        self.is_stable = self.still_count >= STABLE_FRAMES_REQ

    @property
    def stability_ratio(self):
        return min(self.still_count / STABLE_FRAMES_REQ, 1.0)

    @property
    def smoothed_center(self):
        if not self.history:
            return self.center
        return (sum(p[0] for p in self.history) / len(self.history),
                sum(p[1] for p in self.history) / len(self.history))


def select_best_car(cars, zone=None):
    if not cars:
        return None
    def score(box):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if zone is None:
            return area
        zx = zone.polygon_px[:, 0].mean()
        zy = zone.polygon_px[:, 1].mean()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = math.sqrt((cx - zx)**2 + (cy - zy)**2)
        return area / (1 + d * 0.08)
    return max(cars, key=score)