"""Unit tests for FocusStateMachine."""

import pytest

from focus_defence.config import FocusConfig
from focus_defence.state import FocusState, FocusStateMachine


def make_machine(**kwargs) -> FocusStateMachine:
    defaults = {
        "confirm_sec": 0.5,
        "grace_sec": 2.0,
        "persistence_sec": 1.0,
        "escalate_sec": 5.0,
        "away_timeout_sec": 3.0,
        "detect_person": True,
    }
    defaults.update(kwargs)
    return FocusStateMachine(FocusConfig(**defaults), now=0.0, wall_now=1000.0)


def test_single_frame_glitch_does_not_trigger_distraction() -> None:
    sm = make_machine()
    snap = sm.update(phone_detected=True, person_detected=True, now=0.1)
    assert snap.state == FocusState.FOCUSED

    # Phone disappears before confirm_sec (0.5s)
    snap = sm.update(phone_detected=False, person_detected=True, now=0.3)
    assert snap.state == FocusState.FOCUSED
    assert snap.distraction_count == 0


def test_grace_period_recovery_does_not_penalize_score() -> None:
    sm = make_machine()
    sm.update(phone_detected=True, person_detected=True, now=0.1)
    snap = sm.update(phone_detected=True, person_detected=True, now=0.7)
    assert snap.state == FocusState.GRACE
    assert snap.distraction_count == 0

    # Put phone away before 2.0s grace expires; wait out persistence_sec (1.0s)
    sm.update(phone_detected=False, person_detected=True, now=1.2)
    snap = sm.update(phone_detected=False, person_detected=True, now=2.0)
    assert snap.state == FocusState.FOCUSED
    assert snap.distraction_count == 0
    assert snap.distracted_sec == 0.0


def test_confirmed_distraction_persistence_escalation_and_recovery() -> None:
    sm = make_machine()
    # Confirm (0.5s) + Grace (2.0s)
    sm.update(phone_detected=True, person_detected=True, now=0.1)
    sm.update(phone_detected=True, person_detected=True, now=0.7)
    sm.update(phone_detected=True, person_detected=True, now=1.7)
    snap = sm.update(phone_detected=True, person_detected=True, now=2.8)
    assert snap.state == FocusState.DISTRACTED
    assert snap.distraction_count == 1
    assert "entered_distracted" in snap.events

    # Brief 0.4s disappearance should stay DISTRACTED (persistence_sec = 1.0s)
    snap = sm.update(phone_detected=False, person_detected=True, now=3.2)
    assert snap.state == FocusState.DISTRACTED
    assert snap.distraction_count == 1

    # Phone back, hold past escalate_sec (5.0s from 2.8 => 7.8s)
    sm.update(phone_detected=True, person_detected=True, now=4.0)
    snap = sm.update(phone_detected=True, person_detected=True, now=8.0)
    assert snap.state == FocusState.DISTRACTED
    assert snap.alert_level == 3
    assert "escalated" in snap.events

    # Put away for >= persistence_sec
    sm.update(phone_detected=False, person_detected=True, now=8.1)
    snap = sm.update(phone_detected=False, person_detected=True, now=9.2)
    assert snap.state == FocusState.FOCUSED
    assert "recovered_focus" in snap.events
    assert len(sm.completed_incidents) == 1
    assert sm.completed_incidents[0].escalated is True
    assert sm.completed_incidents[0].duration_sec > 5.0


def test_away_from_desk_pauses_focus_timer() -> None:
    sm = make_machine(away_timeout_sec=3.0)
    sm.update(phone_detected=False, person_detected=True, now=1.0)
    sm.update(phone_detected=False, person_detected=False, now=2.5)
    snap = sm.update(phone_detected=False, person_detected=False, now=4.2)
    assert snap.state == FocusState.AWAY
    assert "entered_away" in snap.events

    # Time spent while AWAY accrues to away_sec, not focused_sec
    snap = sm.update(phone_detected=False, person_detected=False, now=6.0)
    assert snap.away_sec == pytest.approx(1.8)
    assert snap.focused_sec < 4.5

    snap = sm.update(phone_detected=False, person_detected=True, now=6.5)
    assert snap.state == FocusState.FOCUSED
    assert "returned_from_away" in snap.events


def test_pomodoro_work_to_break_allows_phone() -> None:
    sm = make_machine(pomodoro_enabled=True, work_minutes=0.1, break_minutes=0.1)  # 6s work, 6s break
    for t in [1.5, 3.0, 4.5, 6.2]:
        snap = sm.update(phone_detected=False, person_detected=True, now=t)
    assert snap.state == FocusState.BREAK
    assert "break_started" in snap.events

    # Using phone during break does NOT trigger distraction
    snap = sm.update(phone_detected=True, person_detected=True, now=8.0)
    snap = sm.update(phone_detected=True, person_detected=True, now=10.0)
    assert snap.state == FocusState.BREAK
    assert snap.distraction_count == 0

    # Break finishes after 6s (step <= MAX_DT_SEC)
    sm.update(phone_detected=False, person_detected=True, now=11.5)
    snap = sm.update(phone_detected=False, person_detected=True, now=12.5)
    assert snap.state == FocusState.FOCUSED
    assert snap.pomodoro_cycles_completed == 1
    assert "break_ended" in snap.events


def test_dt_clamped_across_system_sleep() -> None:
    sm = make_machine()
    snap = sm.update(phone_detected=False, person_detected=True, now=3600.0)
    assert snap.focused_sec == sm.MAX_DT_SEC
