"""Unit tests for config loading, detection context filtering, and UI/tray rendering."""

from pathlib import Path
import numpy as np

from focus_defence.config import FocusConfig, load_config
from focus_defence.detector import CLASS_CELL_PHONE, CLASS_PERSON, enhance_for_ai, filter_detections
from focus_defence.state import FocusState, FocusStateMachine
from focus_defence.tray import create_status_icon
from focus_defence.ui import format_duration, render_hud
import cv2


def test_config_toml_and_cli_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "focus.toml"
    cfg_file.write_text(
        '[detection]\nconfidence_threshold = 0.30\n[timings]\ngrace_sec = 4.0\n',
        encoding="utf-8",
    )
    cfg, args = load_config(["--config", str(cfg_file), "--conf", "0.18", "--headless"])
    assert cfg.confidence_threshold == 0.18
    assert cfg.grace_sec == 4.0
    assert cfg.headless is True
    assert args.stats is False

    new_val = cfg.adjust_sensitivity(-0.50)
    assert new_val == 0.05


def test_filter_detections_near_person_ignores_desk_phone() -> None:
    raw = [
        (100, 50, 300, 400, 0.85, CLASS_PERSON),
        (180, 200, 230, 290, 0.55, CLASS_CELL_PHONE),  # held by person
        (550, 350, 610, 420, 0.65, CLASS_CELL_PHONE),  # lying far away on desk
    ]
    res = filter_detections(
        raw_boxes=raw,
        phone_conf=0.22,
        person_conf=0.35,
        detect_person=True,
        require_near_person=True,
    )
    assert res.person_detected is True
    assert res.active_phone_detected is True
    assert len(res.phone_boxes) == 2
    assert res.phone_boxes[0].active is True
    assert res.phone_boxes[1].active is False
    assert res.phone_boxes[1].label == "Phone (Desk)"


def test_enhance_render_hud_and_tray_icon_smoke() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = enhance_for_ai(frame, clahe)
    assert enhanced.shape == frame.shape

    cfg = FocusConfig()
    sm = FocusStateMachine(cfg, now=0.0)
    snap = sm.update(phone_detected=False, person_detected=True, now=1.0)
    det = filter_detections([], 0.22, 0.35, True, False)

    hud = render_hud(frame, det, snap, cfg, sound_enabled=False, now=1.0)
    assert hud.shape == frame.shape
    assert format_duration(3661) == "01:01:01"

    icon_img = create_status_icon(FocusState.FOCUSED)
    assert icon_img.size == (64, 64)

