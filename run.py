"""
Ultralytics YOLO OBB + OpenVINO: Parking zone monitor
Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python openvino_video_inference.py --model yolov8n-obb.pt --source video.mp4 --zones zones.json

zones.json:
    [ { "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }, ... ]

Keys: Q / ESC = quit  |  click CPU / GPU / NPU to switch device

NOTE: yolov8n-obb.pt is trained on DOTA (aerial imagery).
      Its vehicle classes are "small-vehicle" and "large-vehicle".
      If you use a custom model, edit VEHICLE_CLASS_NAMES below.
      On startup the terminal prints ALL class names your model has.
"""

import cv2
import numpy as np
import time
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# ★ Vehicle class names — yolov8n-obb.pt uses DOTA names ★
# If your model uses different names, add them here.
# Run once and check the terminal output to see all available class names.
VEHICLE_CLASS_NAMES = {
    # DOTA dataset (yolov8n-obb.pt default)
    "small-vehicle", "large-vehicle",
    # COCO (standard yolov8 models)
    "car", "truck", "bus", "motorcycle", "van",
    # Other common names
    "vehicle", "auto", "suv", "pickup", "motorbike",
}
# ──────────────────────────────────────────────────────────────────────────────

DEVICES            = ["intel:cpu", "intel:gpu", "intel:npu"]
BUTTON_W           = 90
BUTTON_H           = 34
BUTTON_MARGIN      = 12
BUTTON_Y           = 12
COLOR_ACTIVE       = (0, 180, 0)
COLOR_INACTIVE     = (60, 60, 60)
COLOR_TEXT         = (255, 255, 255)
COLOR_INFO_BG      = (20, 20, 20)
COLOR_FREE         = (0, 220, 0)
COLOR_OCCUPIED     = (0, 0, 220)
FONT               = cv2.FONT_HERSHEY_SIMPLEX
ZONE_ALPHA         = 0.25
OBB_OVERLAP_THRESH = 0.50   # 50% of vehicle box must overlap zone to count as occupied


# ──────────────────────────────────────────────────────────────────────────────
# Sutherland–Hodgman polygon clipping → intersection area
# ──────────────────────────────────────────────────────────────────────────────
def _clip_edge(poly, p1, p2):
    if len(poly) == 0:
        return poly
    def inside(pt):
        return (p2[0]-p1[0])*(pt[1]-p1[1]) - (p2[1]-p1[1])*(pt[0]-p1[0]) >= 0
    def intersect(a, b):
        da, dp = b-a, p2-p1
        d = da[0]*dp[1] - da[1]*dp[0]
        if abs(d) < 1e-10: return a
        t = ((p1[0]-a[0])*dp[1] - (p1[1]-a[1])*dp[0]) / d
        return a + t*da
    out, n = [], len(poly)
    for i in range(n):
        c, p = poly[i].astype(np.float32), poly[i-1].astype(np.float32)
        if inside(c):
            if not inside(p): out.append(intersect(p, c))
            out.append(c)
        elif inside(p):
            out.append(intersect(p, c))
    return np.array(out, dtype=np.float32) if out else np.empty((0, 2), dtype=np.float32)


def poly_intersect_area(a, b):
    clipped = a.astype(np.float32)
    for i in range(len(b)):
        clipped = _clip_edge(clipped, b[i].astype(np.float32), b[(i+1) % len(b)].astype(np.float32))
        if len(clipped) == 0: return 0.0
    if len(clipped) < 3: return 0.0
    x, y = clipped[:, 0], clipped[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def poly_area(pts):
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ensure_ccw(pts):
    pts = pts.astype(np.float32)
    return pts[::-1].copy() if np.cross(pts[1:]-pts[0], pts[:-1]-pts[0]).sum() < 0 else pts.copy()


# ──────────────────────────────────────────────────────────────────────────────
# OBB → polygon
# ──────────────────────────────────────────────────────────────────────────────
def obb_xywhr_to_poly(xywhr):
    """[cx, cy, w, h, angle_rad] → (4,2) rotated rectangle polygon."""
    cx, cy, w, h, a = (float(v) for v in xywhr[:5])
    c, s = np.cos(a), np.sin(a)
    corners = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]], dtype=np.float32)
    rot = np.array([[c,-s],[s,c]], dtype=np.float32)
    poly = corners @ rot.T
    poly[:, 0] += cx
    poly[:, 1] += cy
    return poly


def xyxy_to_poly(x1, y1, x2, y2):
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Occupancy check
# ──────────────────────────────────────────────────────────────────────────────
def zone_is_occupied(zone_pts, vehicle_polys):
    zone_ccw = ensure_ccw(zone_pts.astype(np.float32))
    for vp in vehicle_polys:
        vp_ccw = ensure_ccw(vp)
        area = poly_area(vp_ccw)
        if area < 1.0: continue
        if poly_intersect_area(vp_ccw, zone_ccw) / area >= OBB_OVERLAP_THRESH:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────
def draw_zones(frame, zones, occupied):
    ov = frame.copy()
    for pts, occ in zip(zones, occupied):
        cv2.fillPoly(ov, [pts], COLOR_OCCUPIED if occ else COLOR_FREE)
    cv2.addWeighted(ov, ZONE_ALPHA, frame, 1-ZONE_ALPHA, 0, frame)
    for pts, occ in zip(zones, occupied):
        cv2.polylines(frame, [pts], True, COLOR_OCCUPIED if occ else COLOR_FREE, 2)


def draw_counter(frame, free, total):
    label = f"Free spots: {free} / {total}"
    pad = 10
    (tw, th), bl = cv2.getTextSize(label, FONT, 0.7, 2)
    x0, y0 = BUTTON_MARGIN, frame.shape[0] - BUTTON_MARGIN - th - bl - pad*2
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0+tw+pad*2, y0+th+bl+pad*2), (20,20,20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, label, (x0+pad, y0+pad+th),
                FONT, 0.7, COLOR_FREE if free > 0 else COLOR_OCCUPIED, 2, cv2.LINE_AA)


def button_rects(w):
    x, rects = BUTTON_MARGIN, []
    for _ in DEVICES:
        rects.append((x, BUTTON_Y, x+BUTTON_W, BUTTON_Y+BUTTON_H))
        x += BUTTON_W + BUTTON_MARGIN
    return rects


def draw_buttons(frame, active, rects):
    for dev, (x1,y1,x2,y2) in zip(DEVICES, rects):
        col = COLOR_ACTIVE if dev == active else COLOR_INACTIVE
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, -1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,200), 1)
        lbl = dev.split(":")[-1].upper()
        tw, th = cv2.getTextSize(lbl, FONT, 0.55, 1)[0]
        cv2.putText(frame, lbl, (x1+(BUTTON_W-tw)//2, y1+(BUTTON_H+th)//2-2),
                    FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_info(frame, device, inf_ms, fps, n_vehicles, is_obb):
    lines = [
        f"Device  : {device.split(':')[-1].upper()}",
        f"Infer   : {inf_ms:.1f} ms",
        f"FPS     : {fps:.1f}",
        f"Vehicles: {n_vehicles}",
        f"Mode    : {'OBB' if is_obb else 'BOX'}",
    ]
    pad, lh, bw = 8, 20, 220
    bh = pad*2 + lh*len(lines)
    x0, y0 = frame.shape[1]-bw-BUTTON_MARGIN, BUTTON_MARGIN
    ov = frame.copy()
    cv2.rectangle(ov, (x0,y0), (x0+bw,y0+bh), COLOR_INFO_BG, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for j, line in enumerate(lines):
        cv2.putText(frame, line, (x0+pad, y0+pad+lh*(j+1)-4),
                    FONT, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)


def hit_test(x, y, rects):
    for i, (x1,y1,x2,y2) in enumerate(rects):
        if x1<=x<=x2 and y1<=y<=y2: return i
    return -1


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path, device):
    print(f"\n[INFO] Loading {model_path} on {device} ...")
    model = YOLO(model_path)

    print("[INFO] All class names in this model:")
    matched = []
    for cid, cname in sorted(model.names.items()):
        is_vehicle = cname.lower() in VEHICLE_CLASS_NAMES
        tag = "  ← VEHICLE (will be tracked)" if is_vehicle else ""
        print(f"       {cid:3d}: {cname}{tag}")
        if is_vehicle:
            matched.append((cid, cname))

    if not matched:
        print("\n[WARNING] *** No vehicle classes matched VEHICLE_CLASS_NAMES! ***")
        print("[WARNING] Zones will never turn red. Add the correct names to VEHICLE_CLASS_NAMES.")
    else:
        print(f"\n[INFO] Tracking {len(matched)} class(es): {[n for _,n in matched]}")

    model.predict(np.zeros((640,640,3), dtype=np.uint8), device=device, verbose=False)
    print("[INFO] Model ready.\n")
    return model


def get_vehicle_ids(model):
    return {cid for cid, name in model.names.items()
            if name.lower() in VEHICLE_CLASS_NAMES}


def is_obb_result(result):
    return result.obb is not None and result.obb.xywhr is not None and len(result.obb.xywhr) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Mouse state
# ──────────────────────────────────────────────────────────────────────────────
class State:
    clicked_device = None

state = State()

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = hit_test(x, y, param["rects"])
        if idx >= 0:
            state.clicked_device = DEVICES[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def load_zones(path):
    with open(path) as f:
        data = json.load(f)
    return [np.array(e["points"], dtype=np.int32) for e in data]


def main(model_path, source, zones_path, default_device):
    zones = []
    if zones_path and Path(zones_path).exists():
        zones = load_zones(zones_path)
        print(f"[INFO] Loaded {len(zones)} zone(s) from {zones_path}")
    elif zones_path:
        print(f"[WARN] Zones file not found: {zones_path}")

    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    rects = button_rects(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    win = "OpenVINO Parking Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb, {"rects": rects})

    current_device = default_device
    model = load_model(model_path, current_device)
    vehicle_ids = get_vehicle_ids(model)

    inf_ms, fps, t_prev = 0.0, 0.0, time.perf_counter()
    using_obb = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Device switch
        if state.clicked_device and state.clicked_device != current_device:
            current_device = state.clicked_device
            state.clicked_device = None
            model = load_model(model_path, current_device)
            vehicle_ids = get_vehicle_ids(model)
            rects = button_rects(frame.shape[1])
            cv2.setMouseCallback(win, mouse_cb, {"rects": rects})

        # Inference
        t0 = time.perf_counter()
        results = model.predict(frame, device=current_device, verbose=False)
        inf_ms = (time.perf_counter() - t0) * 1000

        now = time.perf_counter()
        fps = 1.0 / max(now - t_prev, 1e-9)
        t_prev = now

        result = results[0]
        using_obb = is_obb_result(result)

        # Extract vehicle polygons
        vehicle_polys = []
        if using_obb:
            xywhr_all = result.obb.xywhr.cpu().numpy()
            cls_all   = result.obb.cls.cpu().numpy()
            for xywhr, cls_id in zip(xywhr_all, cls_all):
                if int(cls_id) in vehicle_ids:
                    vehicle_polys.append(obb_xywhr_to_poly(xywhr))
        else:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) in vehicle_ids:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        vehicle_polys.append(xyxy_to_poly(x1, y1, x2, y2))

        # Draw
        annotated = result.plot()

        if zones:
            occupied = [zone_is_occupied(pts, vehicle_polys) for pts in zones]
            draw_zones(annotated, zones, occupied)
            draw_counter(annotated, sum(1 for o in occupied if not o), len(zones))

        draw_buttons(annotated, current_device, rects)
        draw_info(annotated, current_device, inf_ms, fps, len(vehicle_polys), using_obb)

        cv2.imshow(win, annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolov8n-obb.pt")
    parser.add_argument("--source", default="0")
    parser.add_argument("--zones",  default="zones.json")
    parser.add_argument("--device", default="intel:cpu", choices=DEVICES)
    args = parser.parse_args()
    main(args.model, args.source, args.zones, args.device)