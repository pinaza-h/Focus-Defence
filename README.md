# Focus-Defence 🛡️📱

An intelligent, real-time computer vision anti-distraction system that detects **cell phone usage** during study or work sessions to protect your deep focus.

---

## 🚀 Features

- **High-Accuracy Phone Detection**: Powered by **YOLOv8 Small (`yolov8s`)** with ~11.2M parameters to reliably detect phones even when partially occluded by hands, held at angles, or in dark cases.
- **CLAHE Shadow & Contrast Enhancement**: Uses Contrast Limited Adaptive Histogram Equalization to automatically brighten shadows and sharpen edges, ensuring reliable phone detection in backlit rooms (e.g. facing windows) and dim lighting.
- **Threaded 30 FPS Architecture**: AI inference runs in a background worker thread, ensuring the camera stream remains buttery smooth at full 30 FPS with zero video lag.
- **False-Positive Elimination**: Strictly filtered to COCO Class 67 (`cell phone`), preventing microphones, remotes, and other desk accessories from triggering false alarms.
- **Temporal Persistence (1.4s)**: Prevents warning alarms from flickering on and off if your grip shifts or a single frame is obscured.
- **Dynamic Distraction Alerts**:
  - 🟢 **Focus Mode**: Clean green HUD tracking continuous uninterrupted focus time.
  - 🔴 **Distracted Mode**: Flashing red perimeter alarm with a real-time counter tracking `Time on phone: Xs`.
- **Focus Analytics**:
  - Live **Focus Score %** (`Net Focus Time / Total Session Time`).
  - Distraction incident counter (`Distractions: X`).
- **Interactive Controls**:
  - On-the-fly sensitivity tuning with hotkeys (`[` and `]`).
  - Toggleable audio warning beep (`m`).

---

## 🎮 Controls

| Key | Action |
|:---:|:---|
| **`[`** | **Increase sensitivity** (lowers confidence threshold by 0.05) |
| **`]`** | **Decrease sensitivity** (raises confidence threshold by 0.05) |
| **`m`** | Toggle audio alarm beep (**Mute / Unmute**) |
| **`r`** | Reset session statistics and focus score |
| **`q`** | Quit application |

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pinaza-h/Focus-Defence.git
   cd Focus-Defence
   ```

2. **Sync dependencies with `uv`**:
   ```bash
   uv sync
   ```

3. **Launch Focus-Defence**:
   ```bash
   uv run main.py
   ```

---

## 📦 Tech Stack

- **Python 3.12**
- **Ultralytics YOLOv8**: Real-time object detection optimized for cell phones.
- **OpenCV**: CLAHE image enhancement, frame capture, and real-time HUD rendering.
- **Threading**: Multi-threaded concurrency for decoupling AI inference from display FPS.
- **uv**: Lightning-fast Python package and environment management.
