import math
import time
import cv2
import mediapipe as mp
from ultralytics import YOLO

# 1. Initialize YOLOv8 for cell phone detection
phone_model = YOLO("yolov8n.pt")
PHONE_CLASS_ID = 67  # 'cell phone' class in COCO dataset
YOLO_INTERVAL = 2    # Run YOLO every 2 frames for optimal FPS
frame_counter = 0
phone_boxes = []

# 2. Initialize MediaPipe Face Mesh with refined landmarks for iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices for eyes
RIGHT_IRIS_CENTER = 468
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

LEFT_IRIS_CENTER = 473
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

webcam = cv2.VideoCapture(0)

# EMA smoothing for iris tracking
smoothed_h_ratio = 0.5
smoothed_v_ratio = 0.5
ALPHA = 0.35

# Calibration state (captures natural resting center gaze)
calibrated = False
calib_samples_h = []
calib_samples_v = []
CALIBRATION_FRAMES = 30
baseline_h = 0.5
baseline_v = 0.5

# Sensitivity thresholds
H_THRESHOLD = 0.05
V_THRESHOLD = 0.04

# Distraction timer state
away_start_time = None
AWAY_TOLERANCE_SEC = 2.0  # Seconds allowed looking away before alert

def get_landmark_pt(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def calculate_eye_ratios(iris, corner1, corner2):
    if corner1[0] <= corner2[0]:
        c_left, c_right = corner1, corner2
    else:
        c_left, c_right = corner2, corner1

    ux = c_right[0] - c_left[0]
    uy = c_right[1] - c_left[1]
    eye_length = math.hypot(ux, uy)
    if eye_length < 1e-4:
        return 0.5, 0.5

    vx = iris[0] - c_left[0]
    vy = iris[1] - c_left[1]

    # Horizontal projection along eye axis
    h_ratio = (vx * ux + vy * uy) / (eye_length * eye_length)

    # Vertical perpendicular offset from eye-corner baseline
    perp_offset = (ux * vy - uy * vx) / (eye_length * eye_length)
    v_ratio = 0.5 + perp_offset * 3.0

    return h_ratio, v_ratio

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue

    # Flip horizontally for natural mirror feel
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    frame_counter += 1
    current_time = time.time()

    # --- 1. CELL PHONE DETECTION (YOLOv8) ---
    if frame_counter % YOLO_INTERVAL == 0:
        # Fast inference with 320px input size
        yolo_res = phone_model.predict(
            img,
            classes=[PHONE_CLASS_ID],
            conf=0.40,
            imgsz=320,
            verbose=False,
        )
        phone_boxes = []
        for box in yolo_res[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            phone_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))

    # Draw detected phones with red warning boxes
    phone_detected = len(phone_boxes) > 0
    for (px1, py1, px2, py2, pconf) in phone_boxes:
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.rectangle(img, (px1, max(0, py1 - 25)), (px1 + 160, py1), (0, 0, 255), -1)
        cv2.putText(img, f"PHONE {pconf*100:.0f}%", (px1 + 5, py1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # --- 2. FACE & GAZE TRACKING (MediaPipe) ---
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img.flags.writeable = False
    results = face_mesh.process(rgb_img)
    rgb_img.flags.writeable = True

    gaze_label = "No Face"
    delta_h = 0.0
    delta_v = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw Face tracking bounding box
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]
        min_x = max(0, min(x_coords) - 15)
        max_x = min(w, max(x_coords) + 15)
        min_y = max(0, min(y_coords) - 20)
        max_y = min(h, max(y_coords) + 20)

        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.putText(img, "Face", (min_x, max(20, min_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Extract landmarks for eyes
        r_iris = get_landmark_pt(landmarks[RIGHT_IRIS_CENTER], w, h)
        r_outer = get_landmark_pt(landmarks[RIGHT_EYE_OUTER], w, h)
        r_inner = get_landmark_pt(landmarks[RIGHT_EYE_INNER], w, h)

        l_iris = get_landmark_pt(landmarks[LEFT_IRIS_CENTER], w, h)
        l_inner = get_landmark_pt(landmarks[LEFT_EYE_INNER], w, h)
        l_outer = get_landmark_pt(landmarks[LEFT_EYE_OUTER], w, h)

        # Calculate gaze ratios
        r_h_ratio, r_v_ratio = calculate_eye_ratios(r_iris, r_outer, r_inner)
        l_h_ratio, l_v_ratio = calculate_eye_ratios(l_iris, l_inner, l_outer)

        raw_h_ratio = (r_h_ratio + l_h_ratio) / 2.0
        raw_v_ratio = (r_v_ratio + l_v_ratio) / 2.0

        smoothed_h_ratio = ALPHA * raw_h_ratio + (1 - ALPHA) * smoothed_h_ratio
        smoothed_v_ratio = ALPHA * raw_v_ratio + (1 - ALPHA) * smoothed_v_ratio

        # Calibration
        if not calibrated:
            calib_samples_h.append(smoothed_h_ratio)
            calib_samples_v.append(smoothed_v_ratio)
            pct = int((len(calib_samples_h) / CALIBRATION_FRAMES) * 100)
            gaze_label = f"Calibrating ({pct}%)"

            if len(calib_samples_h) >= CALIBRATION_FRAMES:
                baseline_h = sum(calib_samples_h) / len(calib_samples_h)
                baseline_v = sum(calib_samples_v) / len(calib_samples_v)
                calibrated = True
        else:
            delta_h = smoothed_h_ratio - baseline_h
            delta_v = smoothed_v_ratio - baseline_v

            # Horizontal
            if delta_h < -H_THRESHOLD:
                h_dir = "Left"
            elif delta_h > H_THRESHOLD:
                h_dir = "Right"
            else:
                h_dir = "Center"

            # Vertical
            if delta_v < -V_THRESHOLD:
                v_dir = "Up"
            elif delta_v > V_THRESHOLD:
                v_dir = "Down"
            else:
                v_dir = "Center"

            # Dominant direction
            if h_dir != "Center" and v_dir != "Center":
                gaze_label = h_dir if abs(delta_h) >= abs(delta_v) else v_dir
            elif h_dir != "Center":
                gaze_label = h_dir
            elif v_dir != "Center":
                gaze_label = v_dir
            else:
                gaze_label = "Center"

        # Eye annotations
        cv2.line(img, r_outer, r_inner, (0, 255, 0), 1)
        cv2.line(img, l_inner, l_outer, (0, 255, 0), 1)
        cv2.circle(img, r_iris, 3, (255, 255, 0), -1)
        cv2.circle(img, l_iris, 3, (255, 255, 0), -1)
        for pt in [r_outer, r_inner, l_inner, l_outer]:
            cv2.circle(img, pt, 2, (0, 255, 0), -1)

    # --- 3. ANTI-DISTRACTION DECISION ENGINE ---
    flash_border = False

    if not calibrated:
        status_text = f"CALIBRATING GAZE ({len(calib_samples_h)*100//CALIBRATION_FRAMES}%)"
        status_color = (0, 165, 255)  # Orange
    elif phone_detected and gaze_label == "Down":
        # Critical distraction: User is looking down at a detected phone
        status_text = "DISTRACTED: LOOKING AT PHONE!"
        status_color = (0, 0, 255)  # Red
        flash_border = True
        away_start_time = None
    elif phone_detected:
        # Phone is in view even if eyes momentarily glance forward
        status_text = "WARNING: PHONE IN VIEW (PUT AWAY)"
        status_color = (0, 69, 255)  # Red-Orange
        flash_border = True
        away_start_time = None
    elif gaze_label in ["Left", "Right", "Up", "Down"]:
        # Looking away from screen
        if away_start_time is None:
            away_start_time = current_time
        away_duration = current_time - away_start_time

        if away_duration >= AWAY_TOLERANCE_SEC:
            status_text = f"DISTRACTED: LOOKING AWAY ({away_duration:.1f}s)"
            status_color = (0, 0, 255)  # Red
            flash_border = True
        else:
            status_text = f"Looking Away... ({away_duration:.1f}s)"
            status_color = (0, 215, 255)  # Yellow
    else:
        # Gaze is Center and no phone in sight
        away_start_time = None
        status_text = "FOCUSED"
        status_color = (0, 255, 0)  # Green

    # Flashing alert border around camera feed on active distraction
    if flash_border and int(current_time * 4) % 2 == 0:
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)

    # --- 4. HUD / DASHBOARD OVERLAY ---
    # Top Status Bar
    bar_width = 460
    cv2.rectangle(img, (15, 15), (bar_width, 85), (20, 20, 20), -1)
    cv2.rectangle(img, (15, 15), (bar_width, 85), status_color, 2)

    # Status title
    cv2.putText(img, f"STATUS: {status_text}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Subtitle: Gaze & Phone status
    phone_badge = "DETECTED" if phone_detected else "None"
    phone_badge_color = (0, 0, 255) if phone_detected else (200, 200, 200)
    cv2.putText(img, f"Gaze: {gaze_label}  |  Phone: {phone_badge}", (25, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Bottom helper bar
    if calibrated:
        cv2.putText(img, f"dH: {delta_h:+.2f}  dV: {delta_v:+.2f} | 'c': re-center | 'q': quit",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(img, "Please look straight at the screen to calibrate...",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Anti-Distraction Monitor", img)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        calibrated = False
        calib_samples_h = []
        calib_samples_v = []

webcam.release()
cv2.destroyAllWindows()
