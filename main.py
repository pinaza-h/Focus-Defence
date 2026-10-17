import time
import winsound
import cv2
from ultralytics import YOLO

# ==============================================================================
# Focus-Defence: AI Phone Detection & Anti-Distraction Monitor
# ==============================================================================

# 1. Initialize YOLOv8
# Nano model runs at high FPS while maintaining high accuracy for cell phones
phone_model = YOLO("yolov8n.pt")
PHONE_CLASS_ID = 67  # 'cell phone' class in COCO dataset

# 2. Camera setup
webcam = cv2.VideoCapture(0)

# Detection parameters
CONFIDENCE_THRESHOLD = 0.38  # Balanced to catch angled phones without false positives
YOLO_IMGSZ = 480             # 480px provides great detail for phones with high FPS
DETECTION_INTERVAL = 1       # Run every frame now that eye-tracking CPU load is removed

# Temporal smoothing (prevents warning from flickering if phone is momentarily occluded)
PERSISTENCE_DURATION = 0.5   # Seconds phone warning stays active after last detection
last_phone_seen_time = 0.0

# Focus Defence Session Stats
session_start_time = time.time()
total_distracted_time = 0.0
distraction_count = 0
in_distraction_incident = False
distraction_incident_start = 0.0

# Sound alert toggle (can toggle with 'm' key)
sound_alert_enabled = False
last_beep_time = 0.0

def format_time(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

print("[Focus-Defence] Running. Press 'q' to quit, 'r' to reset stats, 'm' to toggle audio beep.")

frame_counter = 0
cached_boxes = []

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue

    # Flip horizontally for natural mirror feel
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    current_time = time.time()
    frame_counter += 1

    # --- 1. RUN PHONE DETECTION ---
    if frame_counter % DETECTION_INTERVAL == 0:
        results = phone_model.predict(
            img,
            classes=[PHONE_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        cached_boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cached_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))

    # Check detection & update persistence
    instant_phone_found = len(cached_boxes) > 0
    if instant_phone_found:
        last_phone_seen_time = current_time

    # Phone is considered actively distracting if seen recently within persistence window
    phone_active = (current_time - last_phone_seen_time) < PERSISTENCE_DURATION

    # --- 2. SESSION TIMERS & INCIDENTS ---
    if phone_active:
        if not in_distraction_incident:
            in_distraction_incident = True
            distraction_count += 1
            distraction_incident_start = current_time

        incident_duration = current_time - distraction_incident_start
        total_distracted_time += 1.0 / 30.0  # approximate frame dt

        # Optional audio warning beep every 1.5 seconds if enabled
        if sound_alert_enabled and (current_time - last_beep_time > 1.5):
            winsound.Beep(1200, 250)
            last_beep_time = current_time
    else:
        in_distraction_incident = False

    # Total session calculations
    total_session_time = max(current_time - session_start_time, 1.0)
    net_focus_time = max(0.0, total_session_time - total_distracted_time)
    focus_score = int((net_focus_time / total_session_time) * 100)

    # --- 3. DRAW DETECTED PHONE BOXES ---
    for (px1, py1, px2, py2, pconf) in cached_boxes:
        # High-visibility red alert bounding box
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)

        # Label tag
        label = f"PHONE {int(pconf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (px1, max(0, py1 - 25)), (px1 + tw + 10, py1), (0, 0, 255), -1)
        cv2.putText(img, label, (px1 + 5, max(15, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- 4. VISUAL ALERTS & HUD ---
    if phone_active:
        # Flashing red perimeter border when phone is detected
        if int(current_time * 4) % 2 == 0:
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 10)

        # Main Red Warning Banner
        cv2.rectangle(img, (15, 15), (500, 95), (0, 0, 40), -1)
        cv2.rectangle(img, (15, 15), (500, 95), (0, 0, 255), 2)

        cv2.putText(img, "DISTRACTED: PHONE DETECTED!", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img, f"Time on phone: {incident_duration:.1f}s | Put it away!", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    else:
        # Clean Green Focus Banner
        cv2.rectangle(img, (15, 15), (460, 95), (15, 35, 15), -1)
        cv2.rectangle(img, (15, 15), (460, 95), (0, 255, 0), 2)

        cv2.putText(img, "FOCUS MODE ACTIVE", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Current Focus: {format_time(net_focus_time)}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1)

    # --- 5. BOTTOM STATS BAR ---
    sound_indicator = "ON" if sound_alert_enabled else "OFF"
    stats_text = (
        f"Score: {focus_score}% | "
        f"Distractions: {distraction_count} | "
        f"Audio: {sound_indicator} ('m') | "
        f"'r': reset | 'q': quit"
    )
    cv2.rectangle(img, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.putText(img, stats_text, (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    cv2.imshow("Focus-Defence", img)

    # Keyboard Controls
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        # Reset session statistics
        session_start_time = time.time()
        total_distracted_time = 0.0
        distraction_count = 0
        in_distraction_incident = False
    elif key == ord("m"):
        # Toggle audio warning beep
        sound_alert_enabled = not sound_alert_enabled

webcam.release()
cv2.destroyAllWindows()
