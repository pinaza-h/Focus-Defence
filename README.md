# Focus-Defence 🛡️📱

An intelligent, real-time computer vision anti-distraction system that detects **cell phone usage** during study or work sessions to protect your focus.

---

## 🚀 Features

- **Real-Time Phone Detection**: Powered by YOLOv8 (`yolov8n`), detecting smartphones regardless of angle, orientation, or hand grip.
- **De-Flicker Persistence**: Uses temporal smoothing so momentary hand occlusions don't cause flickering alarms.
- **Dynamic Distraction Alerts**:
  - 🟢 **Focus Mode**: Clean green HUD tracking continuous focused time.
  - 🔴 **Distracted Mode**: Flashing red perimeter alarm and real-time timer tracking how long you've been on your phone.
- **Focus Analytics**:
  - Focus Score percentage (`Net Focus Time / Total Session Time`).
  - Distraction incident counter (`Distractions: X`).
- **Optional Audio Beep**: Toggleable alarm sound alerting you when you pick up your phone.

---

## 🎮 Controls

| Key | Action |
|:---:|:---|
| **`m`** | Toggle audio warning beep (Mute / Unmute) |
| **`r`** | Reset session statistics and focus score |
| **`q`** | Quit application |

---

## 🛠️ Installation & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pinaza-h/Focus-Defence.git
   cd Focus-Defence
   ```

2. **Sync dependencies**:
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
- **Ultralytics YOLOv8**: High-speed real-time cell phone object detection.
- **OpenCV**: High-framerate video stream processing and HUD overlay.
- **uv**: Fast Python environment & package management.
