import cv2
import numpy as np
from ultralytics import YOLO

from calibrator   import Calibrator
from car_tracker  import CarState, select_best_car
from evaluator    import Evaluator, _grade_from_score
from parking_zone import (
    ParkingZone,
    set_frame_size,
    _draw_text_badge,
    _draw_dashed_poly,
)
from config import CAR_CLASS_ID, CONE_CLASS_ID, CAR_CONF, CONE_CONF

# ── Models ──────────────────────────────────────────────────────────
car_model  = YOLO("yolov8n.pt")
cone_model = YOLO(
    r"C:\Users\lenovo\Desktop\parking\runs\detect\runs\detect\cones_model4\weights\best.pt"
)

# ── State ───────────────────────────────────────────────────────────
calibrator = Calibrator()
car_state  = CarState()
evaluator  = Evaluator()
zone       = None
phase      = "CALIBRATING"
frame_idx  = 0

cap = cv2.VideoCapture(
    r"C:\Users\lenovo\Desktop\parking\test_parking.mp4"
)
if not cap.isOpened():
    print("ERROR: cannot open video")
    exit()

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
set_frame_size(W, H)
print(f"Video: {W}x{H}   Starting calibration...")


# ══════════════════════════════════════════════════════════════════
#  Drawing utilities
# ══════════════════════════════════════════════════════════════════

def _corner_tick(frame, x, y, color, corner, size=10, thickness=2):
    sx = 1 if corner in (2, 4) else -1
    sy = 1 if corner in (3, 4) else -1
    cv2.line(frame, (x, y), (x - sx * size, y), color, thickness)
    cv2.line(frame, (x, y), (x, y - sy * size), color, thickness)


def draw_car(frame, box, is_stable, grade_color=(255, 255, 255)):
    x1, y1, x2, y2 = box
    color = grade_color if is_stable else (60, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _corner_tick(frame, x1, y1, color, 1)
    _corner_tick(frame, x2, y1, color, 2)
    _corner_tick(frame, x1, y2, color, 3)
    _corner_tick(frame, x2, y2, color, 4)
    _draw_text_badge(frame, "CAR", (x1, y1 - 6), color)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 1)


def draw_cone(frame, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    color = (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.circle(frame, (cx, cy), 4, color, -1)
    _draw_text_badge(frame, "CONE", (cx, y1 - 6), color)


def draw_top_left_hud(frame, grade_text, score, grade_color,
                      is_stable, still_count, stable_req):
    x, y = 14, 14
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 4), (x + 190, y + 130),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, grade_text, (x, y + 32),
                font, 1.1, grade_color, 2, cv2.LINE_AA)

    score_str = f"{score:.0f}" if score is not None else "--"
    cv2.putText(frame, score_str, (x, y + 75),
                font, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "/ 100", (x + 80, y + 75),
                font, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

    bar_x     = x
    bar_y     = y + 90
    bar_w     = 160
    ratio     = min(still_count / max(stable_req, 1), 1.0)
    fill_w    = int(ratio * bar_w)
    bar_color = (0, 230, 80) if is_stable else (60, 180, 255)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 6), (50, 50, 50), -1)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + 6), bar_color, -1)

    stab_text = "STOPPED" if is_stable else "MOVING..."
    cv2.putText(frame, stab_text, (bar_x, bar_y + 20),
                font, 0.42, bar_color, 1, cv2.LINE_AA)


def draw_top_right_hud(frame, phase_text, cur_frame_idx, vote_count, vote_max):
    font    = cv2.FONT_HERSHEY_SIMPLEX
    lines   = [phase_text,
               f"FRAME {cur_frame_idx:05d}",
               f"VOTES {vote_count:03d}/{vote_max:03d}"]
    colors  = [(0, 210, 130), (120, 120, 120), (120, 120, 120)]
    scales  = [0.5, 0.40, 0.40]
    x_right = W - 14
    for i, (line, color, scale) in enumerate(zip(lines, colors, scales)):
        (tw, _), _ = cv2.getTextSize(line, font, scale, 1)
        cv2.putText(frame, line, (x_right - tw, 28 + i * 22),
                    font, scale, color, 1, cv2.LINE_AA)


def draw_bottom_metrics(frame, sb, vote_count):
    if sb is None:
        return
    bar_h   = 68
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - bar_h), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    items = [
        ("CENTROID",  sb.centroid,  50, (0, 210,  80)),
        ("HEADING",   sb.heading,   15, (0, 210,  80)),
        ("CORNERS",   sb.corners,   15, (60, 200, 255)),
        ("BALANCE",   sb.balance,   10, (60, 200, 255)),
        ("COLLISION", sb.collision, 10,
         (30, 30, 200) if sb.collision_flag else (0, 210, 80)),
    ]

    slot_w = W // (len(items) + 1)
    for i, (label, val, max_val, color) in enumerate(items):
        bx = slot_w * i + slot_w // 2
        by = H - bar_h + 12

        (lw, _), _ = cv2.getTextSize(label, font, 0.38, 1)
        cv2.putText(frame, label, (bx - lw // 2, by + 10),
                    font, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

        bar_total_w = slot_w - 16
        ratio       = min(val / max(max_val, 1), 1.0)
        bar_fill    = int(ratio * bar_total_w)
        bbar_y      = by + 18

        cv2.rectangle(frame,
                      (bx - bar_total_w // 2, bbar_y),
                      (bx + bar_total_w // 2, bbar_y + 5),
                      (50, 50, 50), -1)
        if bar_fill > 0:
            cv2.rectangle(frame,
                          (bx - bar_total_w // 2, bbar_y),
                          (bx - bar_total_w // 2 + bar_fill, bbar_y + 5),
                          color, -1)

        val_str = f"{val:.0f}/{max_val}"
        (vw, _), _ = cv2.getTextSize(val_str, font, 0.45, 1)
        cv2.putText(frame, val_str, (bx - vw // 2, bbar_y + 20),
                    font, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    med_x = slot_w * len(items) + slot_w // 2
    if vote_count > 0:
        med = float(np.median(list(evaluator.vote_window)))
        g_text, g_color = _grade_from_score(med)
        (tw, _), _ = cv2.getTextSize(f"{med:.1f}", font, 0.9, 2)
        cv2.putText(frame, f"{med:.1f}",
                    (med_x - tw // 2, H - bar_h + 50),
                    font, 0.9, g_color, 2, cv2.LINE_AA)
        (mw, _), _ = cv2.getTextSize("MEDIAN", font, 0.35, 1)
        cv2.putText(frame, "MEDIAN",
                    (med_x - mw // 2, H - bar_h + 14),
                    font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    if sb.overlap_frac is not None:
        ov_str = f"OVERLAP {sb.overlap_frac * 100:.0f}%"
        (ow, _), _ = cv2.getTextSize(ov_str, font, 0.38, 1)
        cv2.putText(frame, ov_str, (W - ow - 10, H - 8),
                    font, 0.38, (160, 160, 100), 1, cv2.LINE_AA)


def draw_calibration_ui(frame, cone_boxes, progress):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for box in cone_boxes:
        draw_cone(frame, box)

    bar_y  = H - 50
    bar_x  = 30
    bar_w  = W - 60
    fill_w = int(progress * bar_w)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 14), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + 14), (0, 210, 130), -1)

    pct_text = f"CALIBRATING ZONE  {int(progress * 100)}%"
    cv2.putText(frame, pct_text, (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 130), 1,
                cv2.LINE_AA)
    cv2.putText(frame, "Keep all cones visible. No car in scene.",
                (bar_x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
                cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_idx += 1

    # ── PHASE 1: CALIBRATION ──────────────────────────────────────
    if phase == "CALIBRATING":
        cone_results = cone_model(frame, conf=CONE_CONF, verbose=False)
        cone_boxes   = []
        cone_centers = []

        for r in cone_results:
            for box in r.boxes:
                b  = tuple(map(int, box.xyxy[0]))
                cx = (b[0] + b[2]) // 2
                cy = (b[1] + b[3]) // 2
                cone_boxes.append(b)
                cone_centers.append((cx, cy))

        calibrator.update(cone_centers)
        draw_calibration_ui(frame, cone_boxes, calibrator.progress)

        if calibrator.done:
            zone  = calibrator.zone
            phase = "EVALUATING"
            print("Zone locked. Evaluating...")

    # ── PHASE 2: EVALUATION ───────────────────────────────────────
    else:
        zone.draw(frame)

        car_results  = car_model(frame, conf=CAR_CONF, verbose=False)
        cone_results = cone_model(frame, conf=CONE_CONF, verbose=False)

        cars       = []
        live_cones = []
        cone_boxes = []

        for r in car_results:
            for box in r.boxes:
                if int(box.cls[0]) == CAR_CLASS_ID:
                    cars.append(tuple(map(int, box.xyxy[0])))

        for r in cone_results:
            for box in r.boxes:
                b  = tuple(map(int, box.xyxy[0]))
                cx = (b[0] + b[2]) // 2
                cy = (b[1] + b[3]) // 2
                cone_boxes.append(b)
                live_cones.append((cx, cy))

        for cb in cone_boxes:
            draw_cone(frame, cb)

        best_box    = select_best_car(cars, zone)
        grade_text  = "WAITING"
        grade_color = (100, 100, 100)
        score       = None
        sb          = evaluator.last_breakdown

        if best_box is not None:
            car_state.update(best_box)
            active_cones = live_cones if len(live_cones) >= 2 else None

            if car_state.is_stable:
                sb          = evaluator.evaluate_frame(
                    best_box, zone, active_cones)
                evaluator.vote(sb.total)
                grade_text, grade_color = sb.grade()
                score = sb.total
            else:
                grade_text  = "MOVING"
                grade_color = (60, 180, 255)

            draw_car(frame, best_box, car_state.is_stable, grade_color)

        draw_top_left_hud(frame, grade_text, score, grade_color,
                          car_state.is_stable, car_state.still_count, 22)
        draw_top_right_hud(frame, "PHASE 2 - EVALUATING",
                           frame_idx, len(evaluator.vote_window), 60)
        draw_bottom_metrics(frame, sb, len(evaluator.vote_window))

    cv2.imshow("Parking Evaluation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ══════════════════════════════════════════════════════════════════
#  Final result
# ══════════════════════════════════════════════════════════════════

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 44)
print("  FINAL PARKING RESULT")
print("=" * 44)

if zone is None:
    print("FAIL — calibration never completed")
elif not evaluator.vote_window:
    print("FAIL — car never came to a stable stop")
else:
    med_score, grade, _ = evaluator.final_result()
    sb = evaluator.last_breakdown
    print(f"  Score       : {med_score:.1f} / 100")
    print(f"  Grade       : {grade}")
    print(f"  Overlap     : {sb.overlap_frac * 100:.1f}%")
    print(f"  Centroid    : {sb.centroid:.1f} / 50")
    print(f"  Heading     : {sb.heading:.1f} / 15")
    print(f"  Corners     : {sb.corners:.1f} / 15")
    print(f"  Balance     : {sb.balance:.1f} / 10")
    print(f"  Collision   : {sb.collision:.1f} / 10")
    if sb.collision_flag:
        print("\n  *** COLLISION DETECTED - INSTANT FAIL ***")

print("=" * 44)