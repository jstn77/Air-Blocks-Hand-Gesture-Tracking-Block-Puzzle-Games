# AirBlocks — Web Edition

Gesture-controlled block puzzle running in your browser via Flask.

## Project Structure

```
airblocks/
├── app.py               # Flask server + game logic + MediaPipe loop
├── requirements.txt
└── templates/
    └── index.html       # Browser UI (camera top, game canvas bottom)
```

## How it works

| Piece | Role |
|-------|------|
| **Background thread** | Runs the OpenCV + MediaPipe loop at 30 fps |
| **`/camera`** | MJPEG stream — the browser `<img>` points here |
| **`/state`** | Server-Sent Events — pushes JSON game state 30×/s |
| **`<canvas>`** | Browser redraws the board on every SSE message |

No Pygame, no pygame window — everything renders in the browser.

## Setup

```bash
cd airblocks
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

## Controls

| Gesture | Action |
|---------|--------|
| ✊ Closed fist | Grab a block (hover over it first) |
| 🖐 Open hand  | Drop / place the block |
| Move hand     | Aim cursor at the grid |

## Tuning tips

- **`snap_to_grid` origin**: the `origin_x=250, origin_y=10` values map the
  hand-tracking coordinate space onto the 500×500 canvas. Adjust if blocks
  feel off.
- **Block pick-up radius**: change `abs(...) < 60` in the GRAB section.
- **Camera index**: change `cv2.VideoCapture(0)` if you have multiple cameras.
- **Canvas size**: `GRID_SIZE * CELL_SIZE` → 10 × 50 = 500 px by default.
