"""Unit tests for SQLite session and history storage."""

from datetime import datetime, timedelta
from pathlib import Path

from focus_defence.config import FocusConfig
from focus_defence.state import DistractionIncident, FocusStateMachine
from focus_defence.storage import SessionStore


def test_save_and_query_sessions(tmp_path: Path) -> None:
    db_file = tmp_path / "history.db"
    store = SessionStore(db_file)
    sm = FocusStateMachine(FocusConfig(confirm_sec=0.0, grace_sec=0.0, persistence_sec=0.5))

    # Too short session should not be saved
    short_snap = sm.update(phone_detected=False, person_detected=True, now=1.0)
    assert store.save_session(started_at=1000.0, snapshot=short_snap, incidents=[], min_save_sec=5.0) is None

    # Save two sessions on consecutive days ending today
    now_dt = datetime(2026, 11, 12, 15, 0, 0)
    yesterday_ts = (now_dt - timedelta(days=1)).timestamp()
    today_ts = now_dt.timestamp()

    sm.reset(now=0.0, wall_now=yesterday_ts)
    for t in [1.5, 3.0, 4.5, 6.0]:
        sm.update(phone_detected=False, person_detected=True, now=t, wall_now=yesterday_ts + t)
    snap1 = sm.snapshot(now=6.0)
    sid1 = store.save_session(started_at=yesterday_ts, snapshot=snap1, incidents=[], ended_at=yesterday_ts + 6.0)
    assert sid1 is not None

    sm.reset(now=0.0, wall_now=today_ts)
    for t in [1.5, 3.0, 4.5, 6.0]:
        sm.update(phone_detected=False, person_detected=True, now=t, wall_now=today_ts + t)
    snap2 = sm.snapshot(now=6.0)
    incidents = [DistractionIncident(started_at_wall=today_ts + 2.0, duration_sec=3.5, escalated=False)]
    sid2 = store.save_session(started_at=today_ts, snapshot=snap2, incidents=incidents, ended_at=today_ts + 6.0)
    assert sid2 is not None

    recent = store.get_recent_sessions(limit=5)
    assert len(recent) == 2
    assert recent[0].id == sid2

    summary = store.get_summary(now_dt=now_dt)
    assert summary.total_sessions == 2
    assert summary.daily_streak_days == 2
    assert summary.today_focused_sec >= 5.9
    assert summary.week_focused_sec >= 11.8

