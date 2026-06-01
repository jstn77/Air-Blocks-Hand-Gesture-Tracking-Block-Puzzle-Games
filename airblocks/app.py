import os
import cv2
import mediapipe as mp
import pygame
import numpy as np
import json
import time
import random
import threading
from types import SimpleNamespace
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

app = Flask(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
GRID_SIZE = 10
CELL_SIZE = 50
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
MAX_CAMERA_INDEX = int(os.getenv("MAX_CAMERA_INDEX", "5"))
SCREEN_W = GRID_SIZE * CELL_SIZE
SCREEN_H = 620
GRID_ORIGIN_X = 0
GRID_ORIGIN_Y = 0
BLOCK_SLOT_Y = 560
GRAB_MODES = {
    "fist": "Closed Fist",
    "pinch": "Pinch",
}
DEFAULT_GRAB_MODE = os.getenv("GRAB_MODE", "fist").strip().lower()
if DEFAULT_GRAB_MODE not in GRAB_MODES:
    DEFAULT_GRAB_MODE = "fist"
DEFAULT_ROI = {
    "x": int(os.getenv("ROI_X", "80")),
    "y": int(os.getenv("ROI_Y", "40")),
    "w": int(os.getenv("ROI_W", "480")),
    "h": int(os.getenv("ROI_H", "360")),
}

COLORS = {
    "cyan":   "#00E5FF",
    "yellow": "#FFD600",
    "purple": "#CE93D8",
    "green":  "#69F0AE",
    "red":    "#FF5252",
    "blue":   "#448AFF",
    "orange": "#FFAB40",
}

BLOCK_SHAPES = [
    {"name": "I-H",  "matrix": [[1,1,1,1]],             "color": "cyan"},
    {"name": "I-V",  "matrix": [[1],[1],[1],[1]],         "color": "cyan"},
    {"name": "O",    "matrix": [[1,1],[1,1]],             "color": "yellow"},
    {"name": "T",    "matrix": [[1,1,1],[0,1,0]],         "color": "purple"},
    {"name": "L",    "matrix": [[1,0],[1,0],[1,1]],       "color": "orange"},
    {"name": "J",    "matrix": [[0,1],[0,1],[1,1]],       "color": "blue"},
    {"name": "S",    "matrix": [[0,1,1],[1,1,0]],         "color": "green"},
    {"name": "Z",    "matrix": [[1,1,0],[0,1,1]],         "color": "red"},
    {"name": "1x1",  "matrix": [[1]],                     "color": "yellow"},
    {"name": "2x2",  "matrix": [[1,1],[1,1]],             "color": "purple"},
    {"name": "3-H",  "matrix": [[1,1,1]],                 "color": "green"},
    {"name": "3-V",  "matrix": [[1],[1],[1]],             "color": "red"},
    {"name": "Sq3",  "matrix": [[1,1,1],[1,1,1],[1,1,1]],"color": "blue"},
]

# ─── GAME STATE (shared across threads) ───────────────────────────────────────
game_lock = threading.Lock()
game_state = {
    "state":       "PLAYING",
    "grid":        [[0]*GRID_SIZE for _ in range(GRID_SIZE)],
    "score":       0,
    "blocks":      [],
    "held_idx":    -1,
    "cursor_x":    -100,
    "cursor_y":    -100,
    "gesture":     "NONE",
    "fps":         0,
}

# ─── GAME LOGIC HELPERS ───────────────────────────────────────────────────────
def generate_3_options():
    slot_x = [90, SCREEN_W // 2, SCREEN_W - 90]
    choices = random.sample(BLOCK_SHAPES, 3)
    result = []
    for i, s in enumerate(choices):
        result.append({
            "matrix":    [row[:] for row in s["matrix"]],
            "color":     s["color"],
            "is_alive":  True,
            "base_x":    slot_x[i],
            "base_y":    BLOCK_SLOT_Y,
        })
    return result

def snap_to_grid(px, py, origin_x=GRID_ORIGIN_X, origin_y=GRID_ORIGIN_Y):
    col = round((px - origin_x) / CELL_SIZE)
    row = round((py - origin_y) / CELL_SIZE)
    return row, col

def is_placement_valid(grid, matrix, start_row, start_col):
    for r, row in enumerate(matrix):
        for c, cell in enumerate(row):
            if cell:
                gr, gc = start_row + r, start_col + c
                if gr < 0 or gr >= GRID_SIZE or gc < 0 or gc >= GRID_SIZE:
                    return False
                if grid[gr][gc] != 0:
                    return False
    return True

def place_block(grid, matrix, start_row, start_col, color):
    for r, row in enumerate(matrix):
        for c, cell in enumerate(row):
            if cell:
                grid[start_row + r][start_col + c] = color

def check_and_clear_lines(grid):
    points = 0
    # Clear full rows
    rows_to_clear = [r for r in range(GRID_SIZE) if all(grid[r][c] != 0 for c in range(GRID_SIZE))]
    for r in rows_to_clear:
        grid.pop(r)
        grid.insert(0, [0]*GRID_SIZE)
        points += 100
    # Clear full cols
    cols_to_clear = [c for c in range(GRID_SIZE) if all(grid[r][c] != 0 for r in range(GRID_SIZE))]
    for c in cols_to_clear:
        for r in range(GRID_SIZE):
            grid[r][c] = 0
        points += 100
    return points

def can_place_any(grid, blocks):
    for b in blocks:
        if not b["is_alive"]:
            continue
        m = b["matrix"]
        for sr in range(GRID_SIZE):
            for sc in range(GRID_SIZE):
                if is_placement_valid(grid, m, sr, sc):
                    return True
    return False

def is_game_over(grid, blocks):
    return not can_place_any(grid, blocks)

# ─── MEDIAPIPE SETUP ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Air-Blocks-Hand-Gesture-Tracking-Block-Puzzle-Games", "hand_landmarker.task")
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing MediaPipe hand landmarker model: {MODEL_PATH}")

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)
CAM_W, CAM_H = 640, 480
cap_lock = threading.Lock()
current_camera_index = CAMERA_INDEX

def create_camera_capture(index):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        cam.release()
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    return cam

cap = create_camera_capture(CAMERA_INDEX)
if cap is None and CAMERA_INDEX != 0:
    cap = create_camera_capture(0)
    current_camera_index = 0
if cap is None:
    raise RuntimeError("Could not open any camera input.")

def get_cursor_position(landmarks):
    x = int(landmarks[9].x * SCREEN_W)
    y = int(landmarks[9].y * SCREEN_H)
    return x, y

def get_finger_open_count(landmarks):
    tips  = [8, 12, 16, 20]
    pips  = [6, 10, 14, 18]
    return sum(1 for t, p in zip(tips, pips)
               if landmarks[t].y < landmarks[p].y)

def is_pinch(landmarks, threshold=0.06):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist_sq = (dx * dx) + (dy * dy)
    return dist_sq <= (threshold * threshold)

def is_three_finger_release(landmarks):
    # Target release pose for pinch mode: thumb + index + middle up,
    # while ring and pinky stay folded.
    thumb_open = abs(landmarks[4].x - landmarks[2].x) > 0.04 or landmarks[4].y < landmarks[3].y
    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_open = landmarks[16].y < landmarks[14].y
    pinky_open = landmarks[20].y < landmarks[18].y
    return thumb_open and index_open and middle_open and (not ring_open) and (not pinky_open)

def get_hand_status(landmarks):
    count = get_finger_open_count(landmarks)
    if is_three_finger_release(landmarks):
        return "THREE"
    if is_pinch(landmarks):
        return "PINCH"
    if count >= 3:
        return "OPEN"
    if count == 0:
        return "CLOSED"
    return "NONE"

def is_grab_active(gesture, grab_mode):
    if grab_mode == "pinch":
        return gesture == "PINCH"
    return gesture == "CLOSED"

def is_release_active(gesture, grab_mode):
    if grab_mode == "pinch":
        return gesture == "THREE"
    return gesture == "OPEN"

def clamp_roi(roi, frame_w, frame_h):
    min_size = 80
    x = max(0, min(int(roi.get("x", 0)), frame_w - min_size))
    y = max(0, min(int(roi.get("y", 0)), frame_h - min_size))
    w = max(min_size, min(int(roi.get("w", frame_w)), frame_w - x))
    h = max(min_size, min(int(roi.get("h", frame_h)), frame_h - y))
    return {"x": x, "y": y, "w": w, "h": h}

def map_landmarks_from_roi(hand_landmarks, roi, frame_w, frame_h):
    mapped = []
    for lm in hand_landmarks:
        fx = (roi["x"] + (lm.x * roi["w"])) / frame_w
        fy = (roi["y"] + (lm.y * roi["h"])) / frame_h
        mapped.append(SimpleNamespace(x=fx, y=fy, z=getattr(lm, "z", 0.0)))
    return mapped

def map_point_to_roi(point, roi, roi_w, roi_h):
    fx = roi["x"] + (point[0] * roi["w"] / roi_w)
    fy = roi["y"] + (point[1] * roi["h"] / roi_h)
    return int(fx), int(fy)

# ─── BACKGROUND GAME LOOP THREAD ──────────────────────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()
gesture_mode_lock = threading.Lock()
current_grab_mode = DEFAULT_GRAB_MODE
roi_lock = threading.Lock()
current_roi = DEFAULT_ROI.copy()

def game_loop():
    global latest_frame
    with game_lock:
        game_state["blocks"] = generate_3_options()

    prev_gesture = "NONE"
    fps_timer    = time.time()
    fps_count    = 0
    fps_val      = 0

    while True:
        with cap_lock:
            success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        with roi_lock:
            roi = clamp_roi(current_roi, frame_w, frame_h)
            current_roi.update(roi)

        roi_rgb = frame_rgb[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]]
        roi_h, roi_w = roi_rgb.shape[:2]
        if roi_w < 2 or roi_h < 2:
            time.sleep(0.01)
            continue
        detect_rgb = cv2.resize(roi_rgb, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_rgb)
        timestamp_ms = int(time.time() * 1000)
        results   = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        cursor_x, cursor_y   = -100, -100
        current_gesture      = "NONE"

        if results.hand_landmarks:
            hand_landmarks = map_landmarks_from_roi(results.hand_landmarks[0], roi, frame_w, frame_h)
            cursor_x, cursor_y   = get_cursor_position(hand_landmarks)
            current_gesture      = get_hand_status(hand_landmarks)

            for lm in hand_landmarks:
                cx = int(lm.x * frame.shape[1])
                cy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            for idx in (0, 1, 2, 3):
                handle_x = roi["x"] if idx in (0, 2) else roi["x"] + roi["w"]
                handle_y = roi["y"] if idx in (0, 1) else roi["y"] + roi["h"]
                cv2.circle(frame, (handle_x, handle_y), 6, (0, 255, 0), -1)

        cv2.rectangle(
            frame,
            (roi["x"], roi["y"]),
            (roi["x"] + roi["w"], roi["y"] + roi["h"]),
            (0, 255, 0),
            2,
        )

        # FPS
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_val   = fps_count
            fps_count = 0
            fps_timer = time.time()

        with game_lock:
            gs = game_state
            with gesture_mode_lock:
                grab_mode = current_grab_mode
            gs["cursor_x"] = cursor_x
            gs["cursor_y"] = cursor_y
            gs["gesture"]  = current_gesture
            gs["fps"]      = fps_val

            if gs["state"] == "PLAYING":
                # GRAB
                if is_grab_active(current_gesture, grab_mode) and gs["held_idx"] == -1:
                    for i, b in enumerate(gs["blocks"]):
                        if b["is_alive"]:
                            if abs(cursor_x - b["base_x"]) < 60 and abs(cursor_y - b["base_y"]) < 60:
                                gs["held_idx"] = i
                                break

                # DROP
                elif is_release_active(current_gesture, grab_mode) and gs["held_idx"] != -1:
                    bd     = gs["blocks"][gs["held_idx"]]
                    matrix = bd["matrix"]
                    off_c  = len(matrix[0]) // 2
                    off_r  = len(matrix)    // 2
                    cr, cc = snap_to_grid(cursor_x, cursor_y)
                    sr, sc = cr - off_r, cc - off_c

                    if is_placement_valid(gs["grid"], matrix, sr, sc):
                        place_block(gs["grid"], matrix, sr, sc, bd["color"])
                        gs["score"] += check_and_clear_lines(gs["grid"])
                        bd["is_alive"] = False

                    gs["held_idx"] = -1

                    if not any(b["is_alive"] for b in gs["blocks"]):
                        gs["blocks"] = generate_3_options()

                    if is_game_over(gs["grid"], gs["blocks"]):
                        gs["state"] = "GAME_OVER"

            elif gs["state"] == "GAME_OVER":
                retry_x = (SCREEN_W // 2) - 120
                retry_y = (SCREEN_H // 2) + 10
                exit_x  = (SCREEN_W // 2) + 20
                exit_y  = (SCREEN_H // 2) + 10

                is_grab_now = is_grab_active(current_gesture, grab_mode)
                was_grab_prev = is_grab_active(prev_gesture, grab_mode)
                if is_grab_now and not was_grab_prev:
                    if retry_x <= cursor_x <= (retry_x + 100) and retry_y <= cursor_y <= (retry_y + 42):
                        gs["grid"]     = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
                        gs["score"]    = 0
                        gs["blocks"]   = generate_3_options()
                        gs["held_idx"] = -1
                        gs["state"]    = "PLAYING"
                    elif exit_x <= cursor_x <= (exit_x + 100) and exit_y <= cursor_y <= (exit_y + 42):
                        pass  # Could signal shutdown; ignored in web mode

        prev_gesture = current_gesture

        # Encode frame for MJPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = buf.tobytes()

        time.sleep(1/30)

threading.Thread(target=game_loop, daemon=True).start()

# ─── FLASK ROUTES ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           grid_size=GRID_SIZE,
                           cell_size=CELL_SIZE,
                           colors=json.dumps(COLORS),
                           screen_w=SCREEN_W,
                           screen_h=SCREEN_H,
                           grid_origin_x=GRID_ORIGIN_X,
                           grid_origin_y=GRID_ORIGIN_Y,
                           block_slot_y=BLOCK_SLOT_Y)

@app.route("/camera")
def camera():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1/30)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def state():
    @stream_with_context
    def generate():
        last = None
        while True:
            with game_lock:
                with roi_lock:
                    roi_snapshot = current_roi.copy()
                snap = json.dumps({
                    "state":    game_state["state"],
                    "grid":     game_state["grid"],
                    "score":    game_state["score"],
                    "blocks":   game_state["blocks"],
                    "held_idx": game_state["held_idx"],
                    "cursor_x": game_state["cursor_x"],
                    "cursor_y": game_state["cursor_y"],
                    "gesture":  game_state["gesture"],
                    "fps":      game_state["fps"],
                    "camera_index": current_camera_index,
                    "grab_mode": current_grab_mode,
                    "roi": roi_snapshot,
                })
            if snap != last:
                yield f"data: {snap}\n\n"
                last = snap
            time.sleep(1/30)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/camera/options")
def camera_options():
    return jsonify({
        "current": current_camera_index,
        "options": [
            {"index": i, "label": f"Camera {i}"}
            for i in range(0, MAX_CAMERA_INDEX + 1)
        ],
    })

@app.route("/camera/select", methods=["POST"])
def camera_select():
    global cap, current_camera_index

    payload = request.get_json(silent=True) or {}
    raw_index = payload.get("index")
    try:
        new_index = int(raw_index)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Invalid camera index."}), 400

    if new_index < 0 or new_index > MAX_CAMERA_INDEX:
        return jsonify({"ok": False, "message": f"Camera index must be between 0 and {MAX_CAMERA_INDEX}."}), 400

    new_cap = create_camera_capture(new_index)
    if new_cap is None:
        return jsonify({"ok": False, "message": f"Camera {new_index} is unavailable."}), 400

    with cap_lock:
        old_cap = cap
        cap = new_cap
        current_camera_index = new_index

    old_cap.release()
    return jsonify({"ok": True, "current": current_camera_index})

@app.route("/gesture/options")
def gesture_options():
    with gesture_mode_lock:
        selected = current_grab_mode
    return jsonify({
        "current": selected,
        "options": [
            {"value": key, "label": label}
            for key, label in GRAB_MODES.items()
        ],
    })

@app.route("/gesture/select", methods=["POST"])
def gesture_select():
    global current_grab_mode

    payload = request.get_json(silent=True) or {}
    raw_mode = str(payload.get("mode", "")).strip().lower()
    if raw_mode not in GRAB_MODES:
        return jsonify({"ok": False, "message": "Invalid gesture mode."}), 400

    with gesture_mode_lock:
        current_grab_mode = raw_mode

    return jsonify({"ok": True, "current": current_grab_mode, "label": GRAB_MODES[current_grab_mode]})

@app.route("/roi/options")
def roi_options():
    with roi_lock:
        roi = current_roi.copy()
    return jsonify({
        "current": roi,
        "limits": {
            "x": {"min": 0, "max": CAM_W - 80},
            "y": {"min": 0, "max": CAM_H - 80},
            "w": {"min": 80, "max": CAM_W},
            "h": {"min": 80, "max": CAM_H},
        },
    })

@app.route("/roi/select", methods=["POST"])
def roi_select():
    payload = request.get_json(silent=True) or {}
    try:
        roi_candidate = {
            "x": int(payload.get("x", current_roi["x"])),
            "y": int(payload.get("y", current_roi["y"])),
            "w": int(payload.get("w", current_roi["w"])),
            "h": int(payload.get("h", current_roi["h"])),
        }
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Invalid ROI values."}), 400

    safe_roi = clamp_roi(roi_candidate, CAM_W, CAM_H)
    with roi_lock:
        current_roi.update(safe_roi)

    return jsonify({"ok": True, "current": safe_roi})

if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
