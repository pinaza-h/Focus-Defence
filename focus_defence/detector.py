"""Threaded webcam capture and YOLO + CLAHE phone/person detector."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import FocusConfig

CLASS_PERSON = 0
CLASS_CELL_PHONE = 67


@dataclass(frozen=True)
class DetectionBox:
    """Single object bounding box in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str
    active: bool = True


@dataclass(frozen=True)
class DetectionResult:
    """Latest thread-safe detection output from the background AI worker."""

    phone_boxes: list[DetectionBox] = field(default_factory=list)
    person_boxes: list[DetectionBox] = field(default_factory=list)
    active_phone_detected: bool = False
    person_detected: bool = True
    ai_fps: float = 0.0
    updated_at: float = 0.0


def enhance_for_ai(frame: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """Apply CLAHE luminance equalization to improve detection in dim or backlit rooms."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_eq = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def is_box_near_any(
    target: DetectionBox,
    anchors: list[DetectionBox],
    margin_ratio: float = 0.15,
) -> bool:
    """Return True if target overlaps any anchor box expanded by margin_ratio."""
    for anchor in anchors:
        aw = max(1, anchor.x2 - anchor.x1)
        ah = max(1, anchor.y2 - anchor.y1)
        pad_x = int(aw * margin_ratio)
        pad_y = int(ah * margin_ratio)

        ax1 = anchor.x1 - pad_x
        ay1 = anchor.y1 - pad_y
        ax2 = anchor.x2 + pad_x
        ay2 = anchor.y2 + pad_y

        if target.x1 <= ax2 and target.x2 >= ax1 and target.y1 <= ay2 and target.y2 >= ay1:
            return True
    return False


def filter_detections(
    raw_boxes: list[tuple[int, int, int, int, float, int]],
    phone_conf: float,
    person_conf: float,
    detect_person: bool,
    require_near_person: bool,
) -> DetectionResult:
    """Split raw YOLO detections into person and phone boxes with context filtering."""
    persons: list[DetectionBox] = []
    candidate_phones: list[DetectionBox] = []

    for x1, y1, x2, y2, conf, cls_id in raw_boxes:
        if cls_id == CLASS_PERSON and detect_person and conf >= person_conf:
            persons.append(
                DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=cls_id, label="Person")
            )
        elif cls_id == CLASS_CELL_PHONE and conf >= phone_conf:
            candidate_phones.append(
                DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=cls_id, label="Phone")
            )

    phones: list[DetectionBox] = []
    any_active_phone = False
    for phone in candidate_phones:
        active = True
        label = "Phone"
        if require_near_person and detect_person:
            active = bool(persons) and is_box_near_any(phone, persons)
            if not active:
                label = "Phone (Desk)"
        if active:
            any_active_phone = True
        phones.append(
            DetectionBox(
                x1=phone.x1,
                y1=phone.y1,
                x2=phone.x2,
                y2=phone.y2,
                confidence=phone.confidence,
                class_id=phone.class_id,
                label=label,
                active=active,
            )
        )

    person_present = True if not detect_person else len(persons) > 0
    return DetectionResult(
        phone_boxes=phones,
        person_boxes=persons,
        active_phone_detected=any_active_phone,
        person_detected=person_present,
    )


def export_model_to_onnx(model_path: str, imgsz: int = 416) -> str:
    """Export a YOLO .pt model to ONNX format."""
    model = YOLO(model_path)
    exported = model.export(format="onnx", imgsz=imgsz)
    return str(exported)


def open_camera_capture(camera_index: int) -> cv2.VideoCapture:
    """Open webcam with low-latency MJPG 30 FPS settings, preferring DirectShow on Windows."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    return cv2.VideoCapture(camera_index)


class CameraStream:
    """Background webcam reader so UI rendering never stalls on USB camera I/O."""

    def __init__(self, camera_index: int) -> None:
        self.cap = open_camera_capture(camera_index)
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_read_seq = 0
        self._cam_fps = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def is_opened(self) -> bool:
        """Return True if the underlying camera opened successfully."""
        return self.cap.isOpened()

    def start(self) -> None:
        """Begin continuous background frame grabbing."""
        if self._running or not self.cap.isOpened():
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name="focus-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop reader thread and release camera."""
        self._running = False
        self._new_frame_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.cap.release()

    def read_latest(self, timeout: float = 0.1) -> tuple[np.ndarray | None, float]:
        """Wait briefly for the next fresh camera frame and return (frame, cam_fps)."""
        if not self._new_frame_event.wait(timeout=timeout):
            return None, self._cam_fps
        with self._lock:
            self._new_frame_event.clear()
            if self._frame is None or self._seq == self._last_read_seq:
                return None, self._cam_fps
            self._last_read_seq = self._seq
            return self._frame, self._cam_fps

    def _reader_loop(self) -> None:
        last_ts = time.perf_counter()
        ema_fps = 0.0
        while self._running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            flipped = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt = max(1e-4, now - last_ts)
            last_ts = now
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps == 0.0 else (0.85 * ema_fps + 0.15 * inst_fps)

            with self._lock:
                self._frame = flipped
                self._seq += 1
                self._cam_fps = ema_fps
            self._new_frame_event.set()


class PhoneDetector:
    """Background-threaded YOLO detector decoupled from the camera display loop."""

    def __init__(self, config: FocusConfig) -> None:
        self.config = config
        if not torch.cuda.is_available():
            threads = config.cpu_threads or max(1, min(8, (os.cpu_count() or 4) - 2))
            torch.set_num_threads(threads)

        self._model = YOLO(config.model_path)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._result_lock = threading.Lock()
        self._config_lock = threading.Lock()

        self._latest_frame: np.ndarray | None = None
        self._frame_seq: int = 0
        self._last_processed_seq: int = -1
        self._latest_result = DetectionResult()

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background inference thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="focus-ai-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background inference thread cleanly."""
        self._running = False
        self._frame_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit_frame(self, frame: np.ndarray) -> None:
        """Provide the latest camera frame to the background AI worker."""
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_seq += 1
        self._frame_event.set()

    def get_latest_result(self) -> DetectionResult:
        """Read the most recent detection result safely."""
        with self._result_lock:
            return self._latest_result

    def update_confidence(self, new_threshold: float) -> None:
        """Thread-safely update the phone confidence threshold."""
        with self._config_lock:
            self.config.confidence_threshold = new_threshold

    def _worker_loop(self) -> None:
        last_infer_time = 0.0
        ema_fps = 0.0

        while self._running:
            self._frame_event.wait(timeout=0.05)

            frame_to_run: np.ndarray | None = None
            with self._frame_lock:
                self._frame_event.clear()
                if self._latest_frame is not None and self._frame_seq != self._last_processed_seq:
                    frame_to_run = self._latest_frame
                    self._last_processed_seq = self._frame_seq

            if frame_to_run is None:
                continue

            if self.config.ai_fps_limit > 0.0:
                min_interval = 1.0 / self.config.ai_fps_limit
                elapsed_since_last = time.perf_counter() - last_infer_time
                if elapsed_since_last < min_interval:
                    time.sleep(min_interval - elapsed_since_last)

            t0 = time.perf_counter()
            with self._config_lock:
                phone_conf = self.config.confidence_threshold
                person_conf = self.config.person_confidence_threshold
                detect_person = self.config.detect_person
                require_near = self.config.require_near_person
                clahe_on = self.config.clahe_enabled
                imgsz = self.config.imgsz
                device = None if self.config.device == "auto" else self.config.device

            ai_input = enhance_for_ai(frame_to_run, self._clahe) if clahe_on else frame_to_run
            target_classes = [CLASS_PERSON, CLASS_CELL_PHONE] if detect_person else [CLASS_CELL_PHONE]
            min_conf = min(phone_conf, person_conf) if detect_person else phone_conf

            predict_kwargs: dict[str, Any] = {
                "classes": target_classes,
                "conf": min_conf,
                "imgsz": imgsz,
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = device

            results = self._model.predict(ai_input, **predict_kwargs)
            raw_boxes: list[tuple[int, int, int, int, float, int]] = []
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, cls_id))

            parsed = filter_detections(
                raw_boxes=raw_boxes,
                phone_conf=phone_conf,
                person_conf=person_conf,
                detect_person=detect_person,
                require_near_person=require_near,
            )

            now = time.perf_counter()
            infer_dt = max(1e-4, now - t0)
            inst_fps = 1.0 / infer_dt
            ema_fps = inst_fps if ema_fps == 0.0 else (0.75 * ema_fps + 0.25 * inst_fps)
            last_infer_time = now

            final_result = DetectionResult(
                phone_boxes=parsed.phone_boxes,
                person_boxes=parsed.person_boxes,
                active_phone_detected=parsed.active_phone_detected,
                person_detected=parsed.person_detected,
                ai_fps=ema_fps,
                updated_at=now,
            )
            with self._result_lock:
                self._latest_result = final_result
