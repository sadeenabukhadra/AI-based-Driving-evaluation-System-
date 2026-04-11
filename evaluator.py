# evaluator.py
import math
import numpy as np
from collections import deque
from config import (
    CENTROID_WEIGHT, HEADING_WEIGHT, CORNER_WEIGHT,
    BALANCE_WEIGHT, COLLISION_WEIGHT,
    HEADING_TOLERANCE_DEG,
    OVERLAP_FULL_THRESH, OVERLAP_PASS_THRESH,
    COLLISION_CM, TOO_CLOSE_CM, TOO_FAR_CM,
    VOTE_WINDOW, GRADE_PERFECT, GRADE_GOOD, GRADE_ACCEPTABLE
)


class ScoreBreakdown:
    def __init__(self):
        self.centroid       = 0.0
        self.heading        = 0.0
        self.corners        = 0.0
        self.balance        = 0.0
        self.collision      = 0.0
        self.collision_flag = False
        self.overlap_frac   = 0.0
        self.status_text    = ""

    @property
    def total(self):
        if self.collision_flag:
            return 0.0
        return (self.centroid + self.heading + self.corners +
                self.balance + self.collision)

    def grade(self):
        if self.collision_flag:
            return "COLLISION", (0, 50, 220)
        t = self.total
        if t >= GRADE_PERFECT:    return "PERFECT",    (0, 230, 80)
        if t >= GRADE_GOOD:       return "GOOD",       (0, 200, 120)
        if t >= GRADE_ACCEPTABLE: return "ACCEPTABLE", (0, 165, 255)
        return "FAIL", (30, 30, 220)


class Evaluator:
    def __init__(self):
        self.vote_window    = deque(maxlen=VOTE_WINDOW)
        self.last_breakdown = ScoreBreakdown()

    def evaluate_frame(self, car_box, zone, cone_centers_px=None):
        sb = ScoreBreakdown()

        # ── 1. OVERLAP FRACTION (primary, camera-angle immune) ───────
        # Pixel-space mask: what fraction of the car is inside the zone?
        frac = zone.car_overlap_fraction(car_box)
        sb.overlap_frac = frac

        if frac >= OVERLAP_FULL_THRESH:
            sb.centroid = CENTROID_WEIGHT                       # full score
        elif frac >= OVERLAP_PASS_THRESH:
            # Linear interpolation between pass and full threshold
            t = (frac - OVERLAP_PASS_THRESH) / (OVERLAP_FULL_THRESH - OVERLAP_PASS_THRESH)
            sb.centroid = CENTROID_WEIGHT * (0.55 + 0.45 * t)  # 55%–100%
        else:
            # Below pass threshold — still give benefit of the doubt
            t = frac / max(OVERLAP_PASS_THRESH, 0.01)
            sb.centroid = CENTROID_WEIGHT * 0.55 * t            # 0%–55%

        # ── 2. HEADING — very wide tolerance ─────────────────────────
        # Bounding-box aspect ratio gives only a coarse angle estimate,
        # which is easily fooled by camera perspective. We give a wide
        # tolerance so this metric helps without punishing.
        car_angle      = _estimate_heading(car_box)
        diff_deg       = abs(math.degrees(car_angle - zone.axis_angle))
        diff_deg       = min(diff_deg, 180 - diff_deg)

        if diff_deg <= HEADING_TOLERANCE_DEG:
            sb.heading = HEADING_WEIGHT
        else:
            excess     = diff_deg - HEADING_TOLERANCE_DEG
            sb.heading = max(0.0, HEADING_WEIGHT * (1.0 - excess / 50.0))

        # ── 3. CORNERS — soft check using warped cm space ────────────
        corners_cm = zone.warp_box(car_box)
        margins    = zone.corner_margins(corners_cm)
        # A corner is "ok" if it's inside or within 25 cm of the edge
        ok_count   = sum(1 for m in margins if m >= -25)
        sb.corners = CORNER_WEIGHT * ((ok_count / 4.0) ** 0.6)

        # ── 4. CONE DISTANCE BALANCE ─────────────────────────────────
        centroid_cm  = corners_cm.mean(axis=0)
        active_cones = cone_centers_px or zone.cone_centers_px

        if active_cones and len(active_cones) >= 2:
            dists_cm = sorted([
                float(np.linalg.norm(centroid_cm - zone.warp_point(cp)))
                for cp in active_cones
            ])
            d1, d2  = dists_cm[0], dists_cm[1]
            avg_d   = (d1 + d2) / 2
            balance = abs(d1 - d2)

            # Collision check
            if d1 < COLLISION_CM:
                sb.collision_flag = True
                sb.status_text    = "COLLISION"
            else:
                sb.collision = COLLISION_WEIGHT

            if avg_d < TOO_CLOSE_CM:
                sb.balance = BALANCE_WEIGHT * 0.5
            elif avg_d > TOO_FAR_CM:
                sb.balance = BALANCE_WEIGHT * 0.6
            elif balance < 20:
                sb.balance = BALANCE_WEIGHT
            elif balance < 50:
                sb.balance = BALANCE_WEIGHT * 0.8
            else:
                sb.balance = max(0, BALANCE_WEIGHT * (1 - (balance - 50) / 100))
        else:
            # No cone data → benefit of the doubt
            sb.balance   = BALANCE_WEIGHT
            sb.collision = COLLISION_WEIGHT

        self.last_breakdown = sb
        return sb

    def vote(self, score_total):
        self.vote_window.append(score_total)

    def final_result(self):
        if not self.vote_window:
            return None, "NO DATA", (128, 128, 128)
        median_score = float(np.median(list(self.vote_window)))
        grade, color = _grade_from_score(median_score)
        return median_score, grade, color


def _estimate_heading(box):
    x1, y1, x2, y2 = box
    return 0.0 if (x2 - x1) >= (y2 - y1) else math.pi / 2


def _grade_from_score(score):
    if score >= GRADE_PERFECT:    return "PERFECT",    (0, 230, 80)
    if score >= GRADE_GOOD:       return "GOOD",       (0, 200, 120)
    if score >= GRADE_ACCEPTABLE: return "ACCEPTABLE", (0, 165, 255)
    return "FAIL", (30, 30, 220)