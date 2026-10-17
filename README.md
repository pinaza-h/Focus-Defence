# Anti-Distraction AI 🎯

A real-time computer vision system that tracks **eye gaze direction** and detects **smartphones** to monitor focus and prevent distraction.

---

## 🚀 Features

- **High-Precision Eye Tracking**: Tracks iris positions using MediaPipe Face Mesh (`refine_landmarks=True`).
- **Tilt-Compensated Baseline**: Uses the skeletal eye-corner axis to accurately detect Up, Down, Left, Right, and Center without eyelid droop interference.
- **1-Second Auto-Calibration**: Captures personal neutral resting gaze on startup; press `c` anytime to recalibrate.
- **Real-Time Phone Detection**: Uses YOLOv8 nano (`yolov8n`) to detect cell phones in real-time.
- **Anti-Distraction Decision Engine**:
  - 🟢 **Focused**: Looking at the screen, no phone in view.
  - 🟡 **Looking Away**: Glancing away from the screen.
  - 🟠 **Phone in View**: Smartphone detected in camera frame.
  - 🔴 **Distracted Alert**: Phone detected **AND** gaze is directed **Down** (flashing red alert border).

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <YOUR_REPO_URL>
   cd "Camera detection"
   ```

2. **Install dependencies with `uv`**:
   ```bash
   uv sync
   ```

3. **Run the application**:
   ```bash
   uv run main.py
   ```

---

## 🎮 Controls

| Key | Action |
|:---:|:---|
| **`c`** | Re-center / recalibrate neutral gaze |
| **`q`** | Quit application |

---

## 📦 Tech Stack

- **Python 3.12**
- **OpenCV**: Camera capture, frame processing, and HUD rendering.
- **MediaPipe**: 478-point 3D Face Mesh with refined iris landmarks.
- **Ultralytics YOLOv8**: Real-time object detection for cell phones.
- **uv**: Fast Python package management.
