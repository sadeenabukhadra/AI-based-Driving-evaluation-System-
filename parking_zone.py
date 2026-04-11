import cv2
import numpy as np
import math
from config import (
    REAL_CONE_DIST_CM,
    ZONE_EXPAND_FRAC_4,
    ZONE_EXPAND_FRAC_3,
    ZONE_EXPAND_FRAC_2,
    SAFE_ZONE_EXTRA,
    ZONE_FLOOR_FRAC,
    ASSUMED_DEPTH_RATIO,
)

FRAME_W = 640
FRAME_H = 480


def set_frame_size(w, h):
    global FRAME_W, FRAME_H
    FRAME_W = int(w)
    FRAME_H = int(h)


class ParkingZone:
    def __init__(self, cone_centers_px, real_cone_dist_cm=REAL_CONE_DIST_CM):
        pts = np.array(cone_centers_px, dtype=np.float32)
        n   = len(pts)

        raw_quad = _build_raw_quad(pts)

        if n >= 4:
            expand_frac = ZONE_EXPAND_FRAC_4
        elif n == 3:
            expand_frac = ZONE_EXPAND_FRAC_3
        else:
            expand_frac = ZONE_EXPAND_FRAC_2

        safe_frac     = expand_frac + SAFE_ZONE_EXTRA
        strict_margin = _compute_margin(raw_quad, expand_frac)
        safe_margin   = _compute_margin(raw_quad, safe_frac)

        self.polygon_px      = _expand_polygon(raw_quad, strict_margin)
        self.safe_polygon_px = _expand_polygon(raw_quad, safe_margin)
        self.cone_centers_px = [tuple(p.astype(int)) for p in pts]

        p0, p1          = raw_quad[0], raw_quad[1]
        px_dist         = float(np.linalg.norm(p0 - p1))
        self.pixel_to_cm = real_cone_dist_cm / max(px_dist, 1e-5)

        w_px = float(np.linalg.norm(self.polygon_px[1] - self.polygon_px[0]))
        h_px = float(np.linalg.norm(self.polygon_px[3] - self.polygon_px[0]))
        self.zone_w_cm = w_px * self.pixel_to_cm
        self.zone_h_cm = h_px * self.pixel_to_cm

        dst = np.array([
            [0,               0              ],
            [self.zone_w_cm,  0              ],
            [self.zone_w_cm,  self.zone_h_cm ],
            [0,               self.zone_h_cm ],
        ], dtype=np.float32)

        self.H     = cv2.getPerspectiveTransform(self.polygon_px, dst)
        self.H_inv = cv2.getPerspectiveTransform(dst, self.polygon_px)

        dx = float(self.polygon_px[1][0] - self.polygon_px[0][0])
        dy = float(self.polygon_px[1][1] - self.polygon_px[0][1])
        self.axis_angle = math.atan2(dy, dx)
        self.center_px  = tuple(self.polygon_px.mean(axis=0).astype(int))

    def warp_point(self, pt_px):
        src = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, self.H)[0][0]

    def warp_box(self, box_px):
        x1, y1, x2, y2 = box_px
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        return cv2.perspectiveTransform(corners, self.H)[0]

    def corner_margins(self, corners_cm):
        return [
            min(x, y, self.zone_w_cm - x, self.zone_h_cm - y)
            for (x, y) in corners_cm
        ]

    def car_overlap_fraction(self, car_box):
        x1, y1, x2, y2 = [int(v) for v in car_box]
        car_w = max(x2 - x1, 1)
        car_h = max(y2 - y1, 1)

        car_mask  = np.zeros((car_h, car_w), dtype=np.uint8)
        zone_mask = np.zeros((car_h, car_w), dtype=np.uint8)
        cv2.rectangle(car_mask, (0, 0), (car_w - 1, car_h - 1), 255, -1)

        shifted = (self.polygon_px - np.array([x1, y1], dtype=np.float32)
                   ).astype(np.int32)
        cv2.fillPoly(zone_mask, [shifted], 255)

        inter = cv2.bitwise_and(car_mask, zone_mask)
        return float(inter.sum()) / float(car_mask.sum() + 1e-5)

    def draw(self, frame):
        _draw_dashed_poly(frame, self.safe_polygon_px,
                          (0, 210, 180), thickness=2, gap=12)
        _draw_text_badge(frame, "SAFE ZONE",
                         _poly_top_center(self.safe_polygon_px, -18),
                         (0, 210, 180))

        pts = self.polygon_px.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 235, 80), 2)
        _draw_text_badge(frame, "PARKING ZONE",
                         _poly_top_center(self.polygon_px, -18),
                         (0, 235, 80))

        cx, cy = self.center_px
        cv2.line(frame,   (cx - 14, cy), (cx + 14, cy), (0, 235, 80), 1)
        cv2.line(frame,   (cx, cy - 14), (cx, cy + 14), (0, 235, 80), 1)
        cv2.circle(frame, (cx, cy), 3, (0, 235, 80), -1)

        for cp in self.cone_centers_px:
            cv2.circle(frame, cp, 6, (0, 200, 255), -1)
            cv2.circle(frame, cp, 9, (0, 200, 255), 1)


def _build_raw_quad(pts):
    n = len(pts)
    if n >= 4:
        return _order_quad(pts[:4])
    if n == 3:
        return _quad_from_3(pts)
    return _quad_from_2(pts)


def _order_quad(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_from_3(pts):
    best_d, best_i, best_j = 0.0, 0, 1
    for i in range(3):
        for j in range(i + 1, 3):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d > best_d:
                best_d, best_i, best_j = d, i, j

    far_a = pts[best_i].copy()
    far_b = pts[best_j].copy()
    lone_idx = [k for k in range(3) if k not in (best_i, best_j)][0]
    lone  = pts[lone_idx].copy()

    cand_a = lone + (far_a - far_b)
    cand_b = lone + (far_b - far_a)

    quad_a = _order_quad(np.array([far_a, far_b, lone, cand_a]))
    quad_b = _order_quad(np.array([far_a, far_b, lone, cand_b]))
    quad   = quad_a if _quad_area(quad_a) >= _quad_area(quad_b) else quad_b
    return _ensure_depth_ratio(quad, ASSUMED_DEPTH_RATIO)


def _quad_from_2(pts):
    p0       = pts[0].astype(float)
    p1       = pts[1].astype(float)
    edge     = p1 - p0
    edge_len = max(float(np.linalg.norm(edge)), 1.0)
    perp     = np.array([-edge[1], edge[0]]) / edge_len
    if perp[1] < 0:
        perp = -perp
    depth = edge_len * ASSUMED_DEPTH_RATIO
    p2 = p1 + perp * depth
    p3 = p0 + perp * depth
    return _order_quad(np.array([p0, p1, p2, p3], dtype=np.float32))


def _ensure_depth_ratio(quad, min_ratio):
    ordered = _order_quad(quad)
    width   = max(float(np.linalg.norm(ordered[1] - ordered[0])), 1.0)
    depth   = float(np.linalg.norm(ordered[3] - ordered[0]))
    if depth / width >= min_ratio:
        return ordered
    scale       = min_ratio / (depth / width)
    mid_top     = (ordered[0] + ordered[1]) / 2
    mid_bot     = (ordered[3] + ordered[2]) / 2
    depth_vec   = mid_bot - mid_top
    new_mid_bot = mid_top + depth_vec * scale
    offset      = new_mid_bot - mid_bot
    ordered[2]  = ordered[2] + offset
    ordered[3]  = ordered[3] + offset
    return ordered


def _quad_area(quad):
    n, area = len(quad), 0.0
    for i in range(n):
        j = (i + 1) % n
        area += quad[i][0] * quad[j][1] - quad[j][0] * quad[i][1]
    return abs(area) / 2.0


def _compute_margin(quad, fraction):
    w     = float(np.linalg.norm(quad[1] - quad[0]))
    h     = float(np.linalg.norm(quad[3] - quad[0]))
    avg   = (w + h) / 2.0
    diag  = math.sqrt(FRAME_W ** 2 + FRAME_H ** 2)
    floor = diag * ZONE_FLOOR_FRAC
    return max(avg * fraction, floor)


def _expand_polygon(poly, margin_px):
    cx = poly[:, 0].mean()
    cy = poly[:, 1].mean()
    out = []
    for pt in poly:
        dx = pt[0] - cx
        dy = pt[1] - cy
        d  = math.sqrt(dx * dx + dy * dy) or 1.0
        out.append([pt[0] + dx / d * margin_px,
                    pt[1] + dy / d * margin_px])
    return np.array(out, dtype=np.float32)


def _draw_dashed_poly(frame, poly, color, thickness=1, gap=10):
    pts = poly.astype(int)
    n   = len(pts)
    for i in range(n):
        _draw_dashed_line(frame, tuple(pts[i]),
                          tuple(pts[(i + 1) % n]), color, thickness, gap)


def _draw_dashed_line(frame, p1, p2, color, thickness=1, gap=10):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L  = math.sqrt(dx * dx + dy * dy)
    if L < 1:
        return
    steps = max(int(L / gap), 1)
    for k in range(0, steps, 2):
        x1 = int(p1[0] + dx * k        / steps)
        y1 = int(p1[1] + dy * k        / steps)
        x2 = int(p1[0] + dx * (k + 1)  / steps)
        y2 = int(p1[1] + dy * (k + 1)  / steps)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def _poly_top_center(poly, offset=0):
    top_pts = sorted(poly, key=lambda p: p[1])[:2]
    cx = int((top_pts[0][0] + top_pts[1][0]) / 2)
    cy = int(min(p[1] for p in top_pts)) + offset
    return (cx, cy)


def _draw_text_badge(frame, text, pos, color):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = pos[0] - tw // 2
    y = pos[1]
    cv2.rectangle(frame, (x - 4, y - th - 3),
                  (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, 1, cv2.LINE_AA)