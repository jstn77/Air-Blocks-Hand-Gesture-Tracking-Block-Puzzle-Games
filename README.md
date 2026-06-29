# 🎮 AirBlocks — Hand-Gesture Block Puzzle

A block-puzzle game you play **with your hands in the air** — no mouse, no
keyboard. A webcam tracks your hand with **MediaPipe**, and you grab and place
Tetris-style blocks on a 10×10 grid by making gestures. Clear full rows or
columns to score.

> Think *Block Blast / Wood Block Puzzle*, but the controller is your hand.

---

## 🌐 Live Demo

> **▶️ Try the game here:** [click on this link](https://huggingface.co/spaces/britod/airblocks-handgesture-games)  
> or watch [this video](https://drive.google.com/file/d/1rzPOcpkYdnbqFuqbE1CfXpa-nOd3SOLO/view) for explanation about how to use the application

---

## ✨ Features

- **Real-time hand tracking** with MediaPipe Hand Landmarker (21 landmarks).
- **Gesture controls** — closed fist (or pinch) to grab, open hand to drop.
- **10×10 puzzle board** with row/column line clears and scoring.
- **Adjustable ROI** so you can frame the exact area your hand moves in.
- **Two grab modes** — Closed Fist or Pinch — switchable on the fly.
- **Camera selector** to pick between multiple connected cameras.
- **Built-in metrics** — average FPS plus grab/drop accuracy, exported to CSV.
- **Two editions**: a Flask server build and a fully client-side browser build.

---

## 🧩 Two Editions

This repo ships the game in two forms:

| Edition | Folder | Where it runs | Best for |
|---------|--------|---------------|----------|
| **Web (Flask server)** | `airblocks/` | Webcam + MediaPipe run on **your machine**; rendered in the browser via MJPEG + SSE | Local play, experiments, metrics |
| **Static (browser-only)** | `docs/` | Webcam + MediaPipe run **entirely in the visitor's browser** — no server | Free deployment (GitHub Pages, HF Spaces, Netlify…) |

The static edition exists because the Flask version reads the camera on the
*server* (`cv2.VideoCapture`), which only works on your own computer — a cloud
host has no camera. The browser edition moves everything client-side so it can
be deployed for free.

---

## 📁 Project Structure

```
Air-Blocks-Hand-Gesture-Tracking-Block-Puzzle-Games/
├── airblocks/                # Flask server edition
│   ├── app.py                # Flask server + game logic + MediaPipe loop
│   ├── requirements.txt      # Python dependencies
│   ├── templates/
│   │   └── index.html        # Browser UI (camera + game canvas)
│   └── README.md
├── docs/                     # Static browser edition (deployable)
│   ├── index.html            # UI (camera canvas + game canvas + controls)
│   ├── app.js                # Camera, MediaPipe (Tasks for Web), game, render
│   ├── .nojekyll
│   └── README.md
├── hand_landmarker.task      # MediaPipe hand-landmark model (~7.8 MB)
├── main.ipynb                # Notebook: evaluation & visualization
└── README.md                 # ← you are here
```

---

## 🚀 Getting Started

### Option A — Flask Server Edition (`airblocks/`)

**Requirements:** Python 3.10+ and a webcam.

```bash
cd airblocks
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

**Optional environment variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAMERA_INDEX` | `1` | Which camera to open first |
| `MAX_CAMERA_INDEX` | `5` | Highest camera index shown in the selector |
| `GRAB_MODE` | `fist` | Grab gesture: `fist` or `pinch` |
| `ROI_X` / `ROI_Y` / `ROI_W` / `ROI_H` | `80 / 40 / 480 / 360` | Region-of-interest box for hand tracking |

### Option B — Static Browser Edition (`docs/`)

Camera access needs a secure context, so `file://` won't work — serve over
`localhost`:

```bash
cd docs
python -m http.server 8000
# open http://localhost:8000
```

Click **START CAMERA** and allow the camera permission. This edition can also
be deployed for free to GitHub Pages, Hugging Face Spaces, Netlify, Cloudflare
Pages, or Vercel — see `docs/README.md` for step-by-step instructions.

---

## 🎯 How to Play

| Gesture | Action |
|---------|--------|
| ✊ Closed fist | Grab a block (hover the cursor over it first) |
| 🖐 Open hand | Drop / place the held block on the grid |
| 🤏 Pinch *(pinch mode)* | 2/3-finger pinch to grab, open to drop |
| 👋 Move hand | Aim the cursor at the grid |

1. Three random blocks appear in the tray at the bottom.
2. Move your hand to hover the cursor over a block, then **grab** it.
3. Move to the grid and **drop** it on a valid spot.
4. Fill an entire **row or column** to clear it and earn points.
5. When no remaining block fits anywhere, it's **game over** — grab the **Retry**
   button to start again.

**Tuning tips:** use the on-screen **ROI** box/sliders to frame where your hand
moves, and the **Grab Gesture** selector to switch between fist and pinch modes.

---

## 🛠️ How It Works (Flask edition)

The server runs two background threads plus the Flask app:

| Component | Role |
|-----------|------|
| **Thread 1 — Camera loop** | Reads the webcam, draws landmarks + ROI, encodes JPEG at ~30 fps |
| **Thread 2 — AI loop** | Runs MediaPipe inference, interprets gestures, updates game state |
| **`/camera`** | MJPEG stream — the browser `<img>` points here |
| **`/state`** | Server-Sent Events — pushes JSON game state ~30×/s |
| **`<canvas>`** | Browser redraws the board on every SSE message |

Gesture detection is geometry-based on the 21 hand landmarks (finger-open
counts, thumb/index pinch distance, etc.), and the cursor is exponentially
smoothed for stability.

---

## 📊 Metrics & Evaluation

The Flask edition tracks experiment metrics during play and writes them to
`experiment_results.csv` on exit (or via the `/save_results` endpoint):

- Average FPS
- Grab attempts, successes, and accuracy
- Drop attempts, successes, and accuracy
- Overall success rate

`main.ipynb` contains the evaluation and visualization workflow for analyzing
these results.

---

## 🧰 Tech Stack

- **Python**, **Flask** — server and game loop
- **OpenCV** — camera capture and frame drawing
- **MediaPipe** (Tasks Vision — Hand Landmarker) — hand tracking
- **NumPy** — array handling
- **HTML5 Canvas + JavaScript** — rendering and UI
- **MediaPipe Tasks for Web** — in-browser tracking (static edition)

---

## 👥 Team Members
### Group 8 LA01 -- COMP7116001 - Computer Vision

| Name | NIM | GitHub |
|------|-----|--------|
| _Brian Nicholas Tedjo_ | 2802403183 |[@britoddd](https://github.com/britoddd) |
| _Jason Budiharjo_ | 2802419446 |[@jason-b123](https://github.com/jason-b123) |
| _Justin Christian Kenan_ | 2802399463 |[@jstn77](https://github.com/jstn77) |
| _Justin Christroper_ | 2802420100 |  |
| _Kian Aurelio Wibowo_ | 2802464582 | [@Kian76-IT](https://github.com/Kian76-IT) |
| _Marvin Adriano Rusdianto_ | 2802402275 | [@Vinn673](https://github.com/Vinn673) |

---

## 📄 License

The Game is published under MIT (see [MIT LICENSE](./LICENSE))
