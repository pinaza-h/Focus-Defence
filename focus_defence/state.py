"""Deterministic state machine for Focus-Defence sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

from .config import FocusConfig


class FocusState(str, Enum):
    """High-level user focus states."""

    FOCUSED = "FOCUSED"
    GRACE = "GRACE"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    PAUSED = "PAUSED"
    BREAK = "BREAK"


@dataclass(frozen=True)
class DistractionIncident:
    """Completed distraction event recorded during a session."""

    started_at_wall: float
    duration_sec: float
    escalated: bool


@dataclass
class SessionSnapshot:
    """Immutable view of current session metrics for UI, tray, and storage."""

    state: FocusState
    focused_sec: float
    distracted_sec: float
    away_sec: float
    paused_sec: float
    break_sec: float
    distraction_count: int
    current_streak_sec: float
    best_streak_sec: float
    current_incident_sec: float
    grace_remaining_sec: float
    alert_level: int
    pomodoro_enabled: bool
    pomodoro_remaining_sec: float
    pomodoro_cycles_completed: int
    events: list[str] = field(default_factory=list)


class FocusStateMachine:
    """Tracks focus states, debouncing, grace periods, streaks, and Pomodoro cycles."""

    MAX_DT_SEC = 2.0

    def __init__(self, config: FocusConfig, now: float | None = None, wall_now: float | None = None) -> None:
        self.config = config
        t = time.monotonic() if now is None else now
        wt = time.time() if wall_now is None else wall_now
        self._reset_fields(t, wt)

    def _reset_fields(self, now: float, wall_now: float) -> None:
        self.session_started_wall: float = wall_now
        self._last_update: float = now

        self.state: FocusState = FocusState.FOCUSED
        self.focused_sec: float = 0.0
        self.distracted_sec: float = 0.0
        self.away_sec: float = 0.0
        self.paused_sec: float = 0.0
        self.break_sec: float = 0.0

        self.distraction_count: int = 0
        self.current_streak_sec: float = 0.0
        self.best_streak_sec: float = 0.0
        self.completed_incidents: list[DistractionIncident] = []

        # Transient timers
        self._phone_confirm_start: float | None = None
        self._grace_active_sec: float = 0.0
        self._last_phone_seen: float = -1e9
        self._last_person_seen: float = now
        self._incident_start: float = 0.0
        self._incident_start_wall: float = wall_now
        self._incident_escalated: bool = False

        # Pomodoro tracking
        self._pomodoro_work_elapsed: float = 0.0
        self._pomodoro_break_elapsed: float = 0.0
        self.pomodoro_cycles_completed: int = 0

    def reset(self, now: float | None = None, wall_now: float | None = None) -> None:
        """Reset all session statistics and return to FOCUSED state."""
        t = time.monotonic() if now is None else now
        wt = time.time() if wall_now is None else wall_now
        self._reset_fields(t, wt)

    def toggle_pause(self, now: float | None = None, wall_now: float | None = None) -> FocusState:
        """Toggle between PAUSED and FOCUSED."""
        t = time.monotonic() if now is None else now
        wt = time.time() if wall_now is None else wall_now
        if self.state == FocusState.PAUSED:
            self.state = FocusState.FOCUSED
            self._last_person_seen = t
            self._phone_confirm_start = None
            self._last_update = t
        else:
            self._finalize_active_incident(t)
            self.state = FocusState.PAUSED
            self._phone_confirm_start = None
            self._grace_active_sec = 0.0
            self._last_update = t
        return self.state

    def toggle_break(self, now: float | None = None) -> FocusState:
        """Start or end a manual/Pomodoro break."""
        t = time.monotonic() if now is None else now
        if self.state == FocusState.BREAK:
            self.state = FocusState.FOCUSED
            self._pomodoro_break_elapsed = 0.0
            self._last_person_seen = t
            self._phone_confirm_start = None
            self._last_update = t
        else:
            self._finalize_active_incident(t)
            self.state = FocusState.BREAK
            self._pomodoro_work_elapsed = 0.0
            self._pomodoro_break_elapsed = 0.0
            self._phone_confirm_start = None
            self._grace_active_sec = 0.0
            self._last_update = t
        return self.state

    def finalize(self, now: float | None = None) -> None:
        """Close any in-progress distraction incident before saving or exiting."""
        t = time.monotonic() if now is None else now
        self._finalize_active_incident(t)

    def _finalize_active_incident(self, now: float) -> None:
        if self.state == FocusState.DISTRACTED:
            duration = max(0.0, now - self._incident_start)
            self.completed_incidents.append(
                DistractionIncident(
                    started_at_wall=self._incident_start_wall,
                    duration_sec=duration,
                    escalated=self._incident_escalated,
                )
            )
            self._incident_escalated = False

    def update(
        self,
        phone_detected: bool,
        person_detected: bool = True,
        now: float | None = None,
        wall_now: float | None = None,
    ) -> SessionSnapshot:
        """Advance the state machine using real elapsed time and latest detections."""
        t = time.monotonic() if now is None else now
        wt = time.time() if wall_now is None else wall_now
        dt = max(0.0, min(t - self._last_update, self.MAX_DT_SEC))
        self._last_update = t

        events: list[str] = []

        # 1. Accrue time for the state we were just in
        self._accrue_time(dt)

        # 2. Update presence / phone tracking timestamps
        person_present = (not self.config.detect_person) or person_detected or phone_detected
        if person_present:
            self._last_person_seen = t
        if phone_detected:
            self._last_phone_seen = t

        # 3. Handle manual PAUSED state (no automatic transitions out)
        if self.state == FocusState.PAUSED:
            return self.snapshot(t, events)

        # 4. Handle BREAK state (Pomodoro timer may auto-end break)
        if self.state == FocusState.BREAK:
            if self.config.pomodoro_enabled:
                break_target = self.config.break_minutes * 60.0
                if self._pomodoro_break_elapsed >= break_target:
                    self._pomodoro_break_elapsed = 0.0
                    self.pomodoro_cycles_completed += 1
                    self.state = FocusState.FOCUSED
                    self._last_person_seen = t
                    self._phone_confirm_start = None
                    events.append("break_ended")
            return self.snapshot(t, events)

        # 5. Active states: FOCUSED, GRACE, DISTRACTED, AWAY
        if self.state == FocusState.AWAY:
            if person_present:
                self.state = FocusState.FOCUSED
                events.append("returned_from_away")
                if phone_detected:
                    self._phone_confirm_start = t
        elif self.state == FocusState.FOCUSED:
            if phone_detected:
                if self._phone_confirm_start is None:
                    self._phone_confirm_start = t
                if (t - self._phone_confirm_start) >= self.config.confirm_sec:
                    self._phone_confirm_start = None
                    if self.config.grace_sec > 0.0:
                        self.state = FocusState.GRACE
                        self._grace_active_sec = 0.0
                        events.append("entered_grace")
                    else:
                        self._enter_distracted(t, wt, events)
            else:
                self._phone_confirm_start = None
                if self.config.detect_person and (t - self._last_person_seen) >= self.config.away_timeout_sec:
                    self.state = FocusState.AWAY
                    events.append("entered_away")
        elif self.state == FocusState.GRACE:
            if phone_detected:
                self._grace_active_sec += dt
                if self._grace_active_sec >= self.config.grace_sec:
                    self._enter_distracted(t, wt, events)
            elif (t - self._last_phone_seen) >= self.config.persistence_sec:
                self._grace_active_sec = 0.0
                self.state = FocusState.FOCUSED
                events.append("recovered_focus")
        elif self.state == FocusState.DISTRACTED:
            incident_elapsed = t - self._incident_start
            if not self._incident_escalated and incident_elapsed >= self.config.escalate_sec:
                self._incident_escalated = True
                events.append("escalated")
            if not phone_detected and (t - self._last_phone_seen) >= self.config.persistence_sec:
                self._finalize_active_incident(t)
                if self.config.detect_person and (t - self._last_person_seen) >= self.config.away_timeout_sec:
                    self.state = FocusState.AWAY
                    events.append("entered_away")
                else:
                    self.state = FocusState.FOCUSED
                    events.append("recovered_focus")

        # 6. Pomodoro work-to-break transition (do not interrupt while paused/away)
        if self.config.pomodoro_enabled and self.state in {
            FocusState.FOCUSED,
            FocusState.GRACE,
            FocusState.DISTRACTED,
        }:
            work_target = self.config.work_minutes * 60.0
            if self._pomodoro_work_elapsed >= work_target:
                self._finalize_active_incident(t)
                self._pomodoro_work_elapsed = 0.0
                self._pomodoro_break_elapsed = 0.0
                self._phone_confirm_start = None
                self._grace_active_sec = 0.0
                self.state = FocusState.BREAK
                events.append("break_started")

        return self.snapshot(t, events)

    def _accrue_time(self, dt: float) -> None:
        if dt <= 0.0:
            return
        if self.state in {FocusState.FOCUSED, FocusState.GRACE}:
            self.focused_sec += dt
            self.current_streak_sec += dt
            self.best_streak_sec = max(self.best_streak_sec, self.current_streak_sec)
            if self.config.pomodoro_enabled:
                self._pomodoro_work_elapsed += dt
        elif self.state == FocusState.DISTRACTED:
            self.distracted_sec += dt
            if self.config.pomodoro_enabled:
                self._pomodoro_work_elapsed += dt
        elif self.state == FocusState.AWAY:
            self.away_sec += dt
        elif self.state == FocusState.PAUSED:
            self.paused_sec += dt
        elif self.state == FocusState.BREAK:
            self.break_sec += dt
            if self.config.pomodoro_enabled:
                self._pomodoro_break_elapsed += dt

    def _enter_distracted(self, now: float, wall_now: float, events: list[str]) -> None:
        self.state = FocusState.DISTRACTED
        self.distraction_count += 1
        self.current_streak_sec = 0.0
        self._incident_start = now
        self._incident_start_wall = wall_now
        self._incident_escalated = False
        self._grace_active_sec = 0.0
        events.append("entered_distracted")

    def snapshot(self, now: float | None = None, events: list[str] | None = None) -> SessionSnapshot:
        """Build a read-only snapshot of the current session state."""
        t = time.monotonic() if now is None else now

        current_incident_sec = 0.0
        grace_remaining_sec = 0.0
        alert_level = 0

        if self.state == FocusState.GRACE:
            grace_remaining_sec = max(0.0, self.config.grace_sec - self._grace_active_sec)
            alert_level = 1
        elif self.state == FocusState.DISTRACTED:
            current_incident_sec = max(0.0, t - self._incident_start)
            alert_level = 3 if current_incident_sec >= self.config.escalate_sec else 2

        pomodoro_remaining = 0.0
        if self.config.pomodoro_enabled:
            if self.state == FocusState.BREAK:
                pomodoro_remaining = max(0.0, self.config.break_minutes * 60.0 - self._pomodoro_break_elapsed)
            else:
                pomodoro_remaining = max(0.0, self.config.work_minutes * 60.0 - self._pomodoro_work_elapsed)

        return SessionSnapshot(
            state=self.state,
            focused_sec=self.focused_sec,
            distracted_sec=self.distracted_sec,
            away_sec=self.away_sec,
            paused_sec=self.paused_sec,
            break_sec=self.break_sec,
            distraction_count=self.distraction_count,
            current_streak_sec=self.current_streak_sec,
            best_streak_sec=self.best_streak_sec,
            current_incident_sec=current_incident_sec,
            grace_remaining_sec=grace_remaining_sec,
            alert_level=alert_level,
            pomodoro_enabled=self.config.pomodoro_enabled,
            pomodoro_remaining_sec=pomodoro_remaining,
            pomodoro_cycles_completed=self.pomodoro_cycles_completed,
            events=list(events) if events else [],
        )

