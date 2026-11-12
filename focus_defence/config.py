"""Configuration loading and validation for Focus-Defence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_CONFIG_FILENAME = "focus_defence.toml"
DEFAULT_DB_PATH = Path.home() / ".focus-defence" / "history.db"


@dataclass
class FocusConfig:
    """Runtime settings for detection, state machine, alerts, and UI."""

    # Camera & model
    camera_index: int = 0
    model_path: str = "yolov8s.pt"
    device: str = "auto"
    imgsz: int = 416
    confidence_threshold: float = 0.22
    person_confidence_threshold: float = 0.35
    ai_fps_limit: float = 0.0
    cpu_threads: int = 0
    clahe_enabled: bool = True

    # Context & state-machine timings (seconds)
    detect_person: bool = True
    require_near_person: bool = False
    confirm_sec: float = 0.5
    persistence_sec: float = 1.4
    grace_sec: float = 2.5
    escalate_sec: float = 10.0
    away_timeout_sec: float = 4.0

    # Pomodoro workflow
    pomodoro_enabled: bool = False
    work_minutes: float = 25.0
    break_minutes: float = 5.0

    # UX, alerts & persistence
    headless: bool = False
    tray_enabled: bool = True
    sound_enabled: bool = False
    notifications_enabled: bool = True
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    min_save_sec: float = 5.0

    def __post_init__(self) -> None:
        self.confidence_threshold = _clamp(self.confidence_threshold, 0.05, 0.90)
        self.person_confidence_threshold = _clamp(self.person_confidence_threshold, 0.10, 0.90)
        self.imgsz = max(160, int(self.imgsz))
        self.ai_fps_limit = max(0.0, float(self.ai_fps_limit))
        self.cpu_threads = max(0, int(self.cpu_threads))
        self.confirm_sec = max(0.0, float(self.confirm_sec))
        self.persistence_sec = max(0.0, float(self.persistence_sec))
        self.grace_sec = max(0.0, float(self.grace_sec))
        self.escalate_sec = max(1.0, float(self.escalate_sec))
        self.away_timeout_sec = max(1.0, float(self.away_timeout_sec))
        self.work_minutes = max(0.1, float(self.work_minutes))
        self.break_minutes = max(0.1, float(self.break_minutes))
        self.db_path = Path(self.db_path).expanduser()

    def adjust_sensitivity(self, delta: float) -> float:
        """Adjust phone confidence threshold within safe bounds and return the new value."""
        self.confidence_threshold = round(_clamp(self.confidence_threshold + delta, 0.05, 0.90), 2)
        return self.confidence_threshold


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def load_toml_overrides(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration key/value overrides from a TOML file if available."""
    candidate = config_path or Path(DEFAULT_CONFIG_FILENAME)
    if not candidate.exists():
        return {}

    with candidate.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict):
        return {}

    # Support either flat keys or sectioned tables ([detection], [timings], etc.)
    flattened: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for Focus-Defence."""
    parser = argparse.ArgumentParser(
        prog="focus-defence",
        description="Real-time AI phone detection and deep-focus monitor.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument("--camera", type=int, default=None, help="Webcam index (default: 0).")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path (.pt or .onnx).")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('auto', 'cpu', 'cuda:0').")
    parser.add_argument("--conf", type=float, default=None, help="Phone confidence threshold (0.05-0.90).")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO input resolution (default: 416).")
    parser.add_argument("--ai-fps", type=float, default=None, help="Max AI inferences/sec (0 = unlimited).")
    parser.add_argument("--grace", type=float, default=None, help="Grace period in seconds before distraction penalty.")
    parser.add_argument("--confirm", type=float, default=None, help="Seconds phone must be seen to trigger grace.")
    parser.add_argument("--persistence", type=float, default=None, help="Seconds to hold state after phone disappears.")
    parser.add_argument("--headless", action="store_true", default=None, help="Start without the OpenCV preview window.")
    parser.add_argument("--no-tray", action="store_true", default=None, help="Disable system tray icon.")
    parser.add_argument("--sound", action="store_true", default=None, help="Enable audio warning beeps on startup.")
    parser.add_argument("--no-notifications", action="store_true", default=None, help="Disable desktop notifications.")
    parser.add_argument("--no-person-check", action="store_true", default=None, help="Disable away-from-desk detection.")
    parser.add_argument(
        "--require-near-person",
        action="store_true",
        default=None,
        help="Ignore phones far from the detected person (reduces desk-phone false positives).",
    )
    parser.add_argument("--pomodoro", action="store_true", default=None, help="Enable Pomodoro work/break cycles.")
    parser.add_argument("--work-min", type=float, default=None, help="Pomodoro focus duration in minutes.")
    parser.add_argument("--break-min", type=float, default=None, help="Pomodoro break duration in minutes.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite history database path.")
    parser.add_argument("--stats", action="store_true", help="Show focus history & analytics and exit.")
    parser.add_argument("--export-onnx", action="store_true", help="Export the configured .pt model to ONNX and exit.")
    return parser


def load_config(argv: list[str] | None = None) -> tuple[FocusConfig, argparse.Namespace]:
    """Build a FocusConfig from defaults, optional TOML file, and CLI flags."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overrides = load_toml_overrides(args.config)
    valid_fields = set(FocusConfig.__dataclass_fields__.keys())
    init_kwargs: dict[str, Any] = {k: v for k, v in overrides.items() if k in valid_fields}

    cli_map: dict[str, Any] = {
        "camera_index": args.camera,
        "model_path": args.model,
        "device": args.device,
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "ai_fps_limit": args.ai_fps,
        "grace_sec": args.grace,
        "confirm_sec": args.confirm,
        "persistence_sec": args.persistence,
        "headless": args.headless,
        "sound_enabled": args.sound,
        "pomodoro_enabled": args.pomodoro,
        "work_minutes": args.work_min,
        "break_minutes": args.break_min,
        "db_path": args.db,
    }
    if args.no_tray:
        cli_map["tray_enabled"] = False
    if args.no_notifications:
        cli_map["notifications_enabled"] = False
    if args.no_person_check:
        cli_map["detect_person"] = False
    if args.require_near_person:
        cli_map["require_near_person"] = True

    for key, val in cli_map.items():
        if val is not None:
            init_kwargs[key] = val

    return FocusConfig(**init_kwargs), args

