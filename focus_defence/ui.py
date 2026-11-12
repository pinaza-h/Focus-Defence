"""OpenCV HUD rendering and terminal analytics formatting."""

from __future__ import annotations

from datetime import datetime
import time

import cv2
import numpy as np

from .config import FocusConfig
from .detector import DetectionBox, DetectionResult
from .state import FocusState, SessionSnapshot
from .storage import HistorySummary, StoredSession

WINDOW_NAME = "Focus-Defence"


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _banner_style(snapshot: SessionSnapshot) -> tuple[str, str, tuple[int, int, int], tuple[int, int, int]]:
    """Return (headline, subtitle, accent_bgr, bg_bgr) for the current focus state."""
    if snapshot.state == FocusState.DISTRACTED:
        headline = "DISTRACTED: PHONE DETECTED!"
        subtitle = f"Time on phone: {snapshot.current_incident_sec:.1f}s | Put it away!"
        return headline, subtitle, (0, 0, 255), (0, 0, 45)

    if snapshot.state == FocusState.GRACE:
        headline = "PHONE SPOTTED - GRACE PERIOD"
        subtitle = f"Put away within {snapshot.grace_remaining_sec:.1f}s to keep streak"
        return headline, subtitle, (0, 190, 255), (15, 35, 50)

    if snapshot.state == FocusState.AWAY:
        headline = "AWAY FROM DESK"
        subtitle = "Timer paused until you return"
        return headline, subtitle, (220, 160, 70), (35, 25, 15)

    if snapshot.state == FocusState.PAUSED:
        headline = "SESSION PAUSED"
        subtitle = "Press 'p' or use Tray to resume"
        return headline, subtitle, (180, 180, 180), (30, 30, 30)

    if snapshot.state == FocusState.BREAK:
        headline = "BREAK TIME - RECHARGE"
        if snapshot.pomodoro_enabled:
            subtitle = f"Break remaining: {format_duration(snapshot.pomodoro_remaining_sec)}"
        else:
            subtitle = "Phone use allowed | Press 'b' to resume focus"
        return headline, subtitle, (255, 200, 0), (45, 35, 10)

    # Default: FOCUSED
    headline = "FOCUS MODE ACTIVE"
    pomo_part = (
        f" | Pomo: {format_duration(snapshot.pomodoro_remaining_sec)}"
        if snapshot.pomodoro_enabled
        else ""
    )
    subtitle = f"Focus: {format_duration(snapshot.focused_sec)} | Streak: {format_duration(snapshot.current_streak_sec)}{pomo_part}"
    return headline, subtitle, (0, 235, 0), (15, 35, 15)


def _draw_box(frame: np.ndarray, box: DetectionBox, color_bgr: tuple[int, int, int], thickness: int) -> None:
    cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color_bgr, thickness)
    tag = f"{box.label} {int(box.confidence * 100)}%"
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top_y = max(0, box.y1 - 24)
    cv2.rectangle(frame, (box.x1, top_y), (box.x1 + tw + 10, box.y1), color_bgr, -1)
    cv2.putText(
        frame,
        tag,
        (box.x1 + 5, max(15, box.y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def render_hud(
    frame: np.ndarray,
    detections: DetectionResult,
    snapshot: SessionSnapshot,
    config: FocusConfig,
    sound_enabled: bool,
    cam_fps: float = 0.0,
    now: float | None = None,
) -> np.ndarray:
    """Draw detections, state banners, and analytics bar onto a copy of the frame."""
    canvas = frame.copy()
    h, w, _ = canvas.shape
    t = time.monotonic() if now is None else now

    # Draw person boxes subtly
    for pbox in detections.person_boxes:
        cv2.rectangle(canvas, (pbox.x1, pbox.y1), (pbox.x2, pbox.y2), (70, 130, 70), 1)

    # Draw phone boxes
    for box in detections.phone_boxes:
        if not box.active:
            color = (140, 140, 140)
        elif snapshot.state == FocusState.GRACE:
            color = (0, 190, 255)
        else:
            color = (0, 0, 255)
        _draw_box(canvas, box, color, thickness=3 if box.active else 1)

    # Flashing perimeter alarm when distracted
    if snapshot.state == FocusState.DISTRACTED and int(t * 4) % 2 == 0:
        border_thickness = 12 if snapshot.alert_level >= 3 else 8
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), border_thickness)

    # Top status banner
    headline, subtitle, accent, bg = _banner_style(snapshot)
    banner_w = min(w - 30, 560)
    cv2.rectangle(canvas, (15, 15), (15 + banner_w, 98), bg, -1)
    cv2.rectangle(canvas, (15, 15), (15 + banner_w, 98), accent, 2)
    cv2.putText(canvas, headline, (26, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, accent, 2)
    cv2.putText(canvas, subtitle, (26, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    # Bottom metrics + hotkeys bar
    sound_ind = "ON" if sound_enabled else "OFF"
    stats_text = (
        f"Focus: {format_duration(snapshot.focused_sec)} | "
        f"Distractions: {snapshot.distraction_count} | "
        f"Best: {format_duration(snapshot.best_streak_sec)} | "
        f"Sens: {config.confidence_threshold:.2f} | "
        f"Sound: {sound_ind} | "
        f"Cam: {cam_fps:.0f}fps AI: {detections.ai_fps:.0f}fps"
    )
    help_text = "Keys: [p]ause [b]reak [v]hide [m]ute ['['/']']sens [r]eset [q]uit"
    cv2.rectangle(canvas, (0, h - 54), (w, h), (15, 15, 15), -1)
    cv2.putText(canvas, stats_text, (14, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(canvas, help_text, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 150), 1)
    return canvas


def format_history_report(summary: HistorySummary, recent: list[StoredSession]) -> str:
    """Format stored focus analytics for terminal output."""
    lines = [
        "",
        "=== Focus-Defence History & Analytics ===",
        f"Total Sessions      : {summary.total_sessions}",
        f"Daily Streak        : {summary.daily_streak_days} day(s)",
        f"Focused Today       : {format_duration(summary.today_focused_sec)}",
        f"Focused Last 7 Days : {format_duration(summary.week_focused_sec)}",
        f"Focused All-Time    : {format_duration(summary.all_time_focused_sec)}",
        f"Best Focus Streak   : {format_duration(summary.best_streak_sec)}",
        f"Total Distractions  : {summary.total_distractions}",
    ]
    if recent:
        lines.append("")
        lines.append("Recent Sessions:")
        for s in recent:
            dt_str = datetime.fromtimestamp(s.started_at).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  - {dt_str} | Focus {format_duration(s.focused_sec)} | "
                f"Distractions {s.distraction_count} | "
                f"Best Streak {format_duration(s.best_streak_sec)}"
            )
    lines.append("")
    return "\n".join(lines)

