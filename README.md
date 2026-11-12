![Uploading Banner.jpg…]()
# Focus-Defence 🛡️📱

An intelligent, privacy-first real-time computer vision anti-distraction system that detects **cell phone usage** during study or work sessions to protect your deep focus.

All video processing runs **100% locally in memory** — zero frames are ever saved or uploaded.

---

## 🚀 Features

- **High-Accuracy Phone & Presence Detection**: Powered by **YOLOv8 Small (`yolov8s`)** at `416px` in a dedicated background thread (`20+ FPS` AI on CPU) alongside a threaded DirectShow/MJPG camera reader for smooth `30 FPS` video.
- **CLAHE Shadow & Contrast Enhancement**: Automatically brightens shadows and sharpens edges for reliable detection in backlit or dim rooms.
- **Context-Aware State Machine**:
  - **Entry Debouncing (`0.5s`)**: Eliminates single-frame false positives from inflating distraction counts.
  - **Grace Period (`2.5s`)**: Gives you time to put a phone away (e.g., checking a 2FA code) before counting a distraction.
  - **Temporal Persistence (`1.4s`)**: Prevents flickering when your grip shifts.
  - **Away-from-Desk Detection**: Detects user presence (`person`) and pauses the focus timer when you step away instead of counting empty-desk time.
  - **Optional Desk-Phone Filter (`--require-near-person`)**: Ignores phones sitting far away on your desk when not near you.
- **Background & System Tray Mode**:
  - Run with `--headless` or press `v` to hide the OpenCV preview window and monitor from the **system tray** with live color-coded status icons and native OS notifications.
- **Pomodoro Mode**: Built-in `--pomodoro` work/break cycles (phone use during breaks doesn't trigger alerts).
- **Non-Blocking Escalating Alerts**: Cross-platform async audio cues (`m` to toggle) and native notifications that escalate if a distraction lasts past `10s`.
- **Persistent Analytics (SQLite)**: Saves sessions, focus time, streaks, and distraction incidents locally (`~/.focus-defence/history.db`). View trends anytime with `--stats`.

---

## 🎮 Controls

| Key | Action |
|:---:|:---|
| **`p`** | **Pause / Resume** session |
| **`b`** | **Start / End Break** (phone allowed) |
| **`v`** | **Hide / Show** camera preview window (runs in tray) |
| **`[`** | **Increase sensitivity** (lowers confidence threshold by 0.05) |
| **`]`** | **Decrease sensitivity** (raises confidence threshold by 0.05) |
| **`m`** | Toggle audio alarm beep (**Mute / Unmute**) |
| **`r`** | Save & reset current session |
| **`q`** | Quit and save session summary |

---

## 🛠️ Installation & Usage

1. **Sync dependencies with `uv`**:
   ```bash
   uv sync
   ```

2. **Launch Focus-Defence**:
   ```bash
   uv run main.py
   ```
   *(Or after `uv sync`, run `uv run focus-defence`.)*

3. **Useful Modes & CLI Flags**:
   ```bash
   # Headless background tray mode with sound & Pomodoro 25/5 cycles
   uv run focus-defence --headless --sound --pomodoro

   # Ignore phones sitting far away on your desk
   uv run focus-defence --require-near-person

   # View focus history, daily streak, and recent sessions
   uv run focus-defence --stats

   # Export yolov8s.pt to ONNX for faster CPU inference
   uv run focus-defence --export-onnx
   ```

4. **Optional Config File**:
   Copy [`focus_defence.toml.example`](focus_defence.toml.example) to `focus_defence.toml` to persist your preferred defaults.

---

## 🧪 Running Tests

```bash
uv run pytest -v
```

---

## 📦 Architecture

- `focus_defence/config.py`: CLI + TOML configuration loading and validation
- `focus_defence/detector.py`: Threaded YOLO + CLAHE detector with context filtering & ONNX export
- `focus_defence/state.py`: Real-time delta-time state machine (debounce, grace, persistence, away, Pomodoro)
- `focus_defence/alerts.py`: Non-blocking cross-platform audio & desktop notifications
- `focus_defence/storage.py`: SQLite session & distraction history store
- `focus_defence/ui.py`: OpenCV HUD & terminal analytics formatting
- `focus_defence/tray.py`: System tray status icon & menu controller
- `focus_defence/app.py`: Main application loop & graceful shutdown
