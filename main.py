import threading
import time
import winsound
import cv2
from ultralytics import YOLO

# ==============================================================================
# Focus-Defence: High-Accuracy Phone Detection & Anti-Distraction Engine
# ==============================================================================

# 1. Load YOLOv8 Small model (drastically better for hand-held & occluded phones)
# yolov8s has ~11M parameters (3.5x more than nano) for recognizing dark cases & hand grips
model = YOLO("yolov8s.pt")

# Target class: strictly 67 = 'cell phone' only (prevents microphones/remotes from triggering)
TARGET_CLASSES = [67]

# Detection settings
confidence_threshold = 0.22   # Tuned to capture dark phones & back of phones in hands
YOLO_IMGSZ = 640              # Full 640px resolution for sharp edge & camera lens detection
PERSISTENCE_SEC = 1.4         # 1.4s smoothing so holding/moving phone never drops detection

# Camera setup
webcam = cv2.VideoCapture(0)

# CLAHE contrast enhancer (helps YOLO detect dark objects in backlit/dim rooms)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_for_ai(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

# Threading state for 30 FPS smooth video + background AI inference
frame_lock = threading.Lock()
boxes_lock = threading.Lock()
latest_frame = None
latest_boxes = []
is_running = True

# Session statistics
session_start_time = time.time()
total_distracted_time = 0.0
distraction_count = 0
in_distraction_incident = False
distraction_incident_start = 0.0
last_phone_seen_time = 0.0

# Audio warning toggle
sound_alert_enabled = False
last_beep_time = 0.0

def detection_worker():
    """Background worker that continuously runs YOLO on the latest camera frame."""
    global latest_boxes
    while is_running:
        with frame_lock:
            if latest_frame is None:
                img_for_yolo = None
            else:
                img_for_yolo = latest_frame.copy()

        if img_for_yolo is not None:
            # Enhance shadows & edge contrast for backlit/dim camera feeds
            ai_input = enhance_for_ai(img_for_yolo)
            results = model.predict(
                ai_input,
                classes=TARGET_CLASSES,
                conf=confidence_threshold,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )
            detected = []
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label_name = "Phone"
                detected.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, label_name))

            with boxes_lock:
                latest_boxes = detected

        time.sleep(0.01)

# Start background AI detection thread
worker_thread = threading.Thread(target=detection_worker, daemon=True)
worker_thread.start()

def format_time(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

print("[Focus-Defence] Running with YOLOv8 Small + Threaded Engine.")
print("Controls: 'm' = toggle sound | '[' = more sensitive | ']' = less sensitive | 'r' = reset | 'q' = quit")

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue

    # Mirror horizontally for natural webcam feel
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    current_time = time.time()

    # Share current frame with background AI worker
    with frame_lock:
        latest_frame = img

    # Read current detections safely from thread
    with boxes_lock:
        active_boxes = list(latest_boxes)

    # Update persistence state
    if len(active_boxes) > 0:
        last_phone_seen_time = current_time

    phone_detected = (current_time - last_phone_seen_time) < PERSISTENCE_SEC

    # Session stats & timing
    if phone_detected:
        if not in_distraction_incident:
            in_distraction_incident = True
            distraction_count += 1
            distraction_incident_start = current_time

        incident_duration = current_time - distraction_incident_start
        total_distracted_time += 1.0 / 30.0

        # Audio warning beep
        if sound_alert_enabled and (current_time - last_beep_time > 1.5):
            winsound.Beep(1200, 250)
            last_beep_time = current_time
    else:
        in_distraction_incident = False

    total_session_time = max(current_time - session_start_time, 1.0)
    net_focus_time = max(0.0, total_session_time - total_distracted_time)
    focus_score = int((net_focus_time / total_session_time) * 100)

    # --- DRAW DETECTED OBJECTS ---
    for (px1, py1, px2, py2, pconf, plabel) in active_boxes:
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)

        tag = f"{plabel} {int(pconf * 100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (px1, max(0, py1 - 25)), (px1 + tw + 10, py1), (0, 0, 255), -1)
        cv2.putText(img, tag, (px1 + 5, max(15, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- VISUAL ALERTS & HUD ---
    if phone_detected:
        # Flashing red perimeter warning on active distraction
        if int(current_time * 4) % 2 == 0:
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 10)

        # Red Alert Banner
        cv2.rectangle(img, (15, 15), (510, 95), (0, 0, 40), -1)
        cv2.rectangle(img, (15, 15), (510, 95), (0, 0, 255), 2)

        cv2.putText(img, "DISTRACTED: PHONE DETECTED!", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img, f"Time on phone: {incident_duration:.1f}s | Put it away!", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    else:
        # Green Focus Banner
        cv2.rectangle(img, (15, 15), (460, 95), (15, 35, 15), -1)
        cv2.rectangle(img, (15, 15), (460, 95), (0, 255, 0), 2)

        cv2.putText(img, "FOCUS MODE ACTIVE", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Current Focus: {format_time(net_focus_time)}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1)

    # Bottom helper & stats bar
    sound_ind = "ON" if sound_alert_enabled else "OFF"
    stats_text = (
        f"Score: {focus_score}% | "
        f"Distractions: {distraction_count} | "
        f"Sens: {confidence_threshold:.2f} ('['/']') | "
        f"Sound: {sound_ind} ('m')"
    )
    cv2.rectangle(img, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.putText(img, stats_text, (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    cv2.imshow("Focus-Defence", img)

    # Key controls
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        session_start_time = time.time()
        total_distracted_time = 0.0
        distraction_count = 0
        in_distraction_incident = False
    elif key == ord("m"):
        sound_alert_enabled = not sound_alert_enabled
    elif key == ord("["):
        # Increase sensitivity (lower confidence threshold)
        confidence_threshold = max(0.10, confidence_threshold - 0.05)
        print(f"[Focus-Defence] Sensitivity adjusted: Conf threshold = {confidence_threshold:.2f}")
    elif key == ord("]"):
        # Decrease sensitivity (raise confidence threshold)
        confidence_threshold = min(0.70, confidence_threshold + 0.05)
        print(f"[Focus-Defence] Sensitivity adjusted: Conf threshold = {confidence_threshold:.2f}")

is_running = False
webcam.release()
cv2.destroyAllWindows()
