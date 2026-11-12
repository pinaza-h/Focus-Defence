"""SQLite session and distraction history persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import time

from .state import DistractionIncident, SessionSnapshot


@dataclass(frozen=True)
class StoredSession:
    """Single saved focus session row."""

    id: int
    started_at: float
    ended_at: float
    focused_sec: float
    distracted_sec: float
    away_sec: float
    break_sec: float
    distraction_count: int
    best_streak_sec: float
    pomodoro_enabled: bool


@dataclass(frozen=True)
class HistorySummary:
    """Aggregated analytics across stored sessions."""

    total_sessions: int
    today_focused_sec: float
    week_focused_sec: float
    all_time_focused_sec: float
    total_distractions: int
    best_streak_sec: float
    daily_streak_days: int


class SessionStore:
    """Persists completed Focus-Defence sessions and incidents to SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    focused_sec REAL NOT NULL,
                    distracted_sec REAL NOT NULL,
                    away_sec REAL NOT NULL,
                    break_sec REAL NOT NULL,
                    distraction_count INTEGER NOT NULL,
                    best_streak_sec REAL NOT NULL,
                    pomodoro_enabled INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS distractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    started_at REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    escalated INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if "focus_score" in cols:
                conn.execute("ALTER TABLE sessions DROP COLUMN focus_score")

    def save_session(
        self,
        started_at: float,
        snapshot: SessionSnapshot,
        incidents: list[DistractionIncident],
        ended_at: float | None = None,
        min_save_sec: float = 5.0,
    ) -> int | None:
        """Save a session and its distraction incidents. Returns session ID or None if too short."""
        active_or_break = snapshot.focused_sec + snapshot.distracted_sec + snapshot.break_sec
        if active_or_break < min_save_sec:
            return None

        end_ts = time.time() if ended_at is None else ended_at
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    started_at, ended_at, focused_sec, distracted_sec,
                    away_sec, break_sec, distraction_count,
                    best_streak_sec, pomodoro_enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    end_ts,
                    round(snapshot.focused_sec, 2),
                    round(snapshot.distracted_sec, 2),
                    round(snapshot.away_sec, 2),
                    round(snapshot.break_sec, 2),
                    snapshot.distraction_count,
                    round(snapshot.best_streak_sec, 2),
                    1 if snapshot.pomodoro_enabled else 0,
                ),
            )
            session_id = int(cursor.lastrowid)

            if incidents:
                conn.executemany(
                    """
                    INSERT INTO distractions (session_id, started_at, duration_sec, escalated)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            inc.started_at_wall,
                            round(inc.duration_sec, 2),
                            1 if inc.escalated else 0,
                        )
                        for inc in incidents
                    ],
                )
        return session_id

    def get_recent_sessions(self, limit: int = 10) -> list[StoredSession]:
        """Return most recent sessions newest-first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, started_at, ended_at, focused_sec, distracted_sec,
                       away_sec, break_sec, distraction_count,
                       best_streak_sec, pomodoro_enabled
                FROM sessions
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            StoredSession(
                id=int(row["id"]),
                started_at=float(row["started_at"]),
                ended_at=float(row["ended_at"]),
                focused_sec=float(row["focused_sec"]),
                distracted_sec=float(row["distracted_sec"]),
                away_sec=float(row["away_sec"]),
                break_sec=float(row["break_sec"]),
                distraction_count=int(row["distraction_count"]),
                best_streak_sec=float(row["best_streak_sec"]),
                pomodoro_enabled=bool(row["pomodoro_enabled"]),
            )
            for row in rows
        ]

    def get_summary(self, now_dt: datetime | None = None) -> HistorySummary:
        """Compute today, 7-day, and all-time focus analytics."""
        ref_dt = datetime.now() if now_dt is None else now_dt
        today_start = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        week_start = (ref_dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6)).timestamp()

        with self._connect() as conn:
            agg = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(focused_sec), 0.0) AS all_focused,
                    COALESCE(SUM(distraction_count), 0) AS total_distractions,
                    COALESCE(MAX(best_streak_sec), 0.0) AS best_streak
                FROM sessions
                """
            ).fetchone()

            today_row = conn.execute(
                "SELECT COALESCE(SUM(focused_sec), 0.0) AS s FROM sessions WHERE started_at >= ?",
                (today_start,),
            ).fetchone()
            week_row = conn.execute(
                "SELECT COALESCE(SUM(focused_sec), 0.0) AS s FROM sessions WHERE started_at >= ?",
                (week_start,),
            ).fetchone()
            timestamps = conn.execute("SELECT started_at FROM sessions ORDER BY started_at DESC").fetchall()

        active_dates = {datetime.fromtimestamp(float(r["started_at"])).date() for r in timestamps}
        daily_streak = _compute_daily_streak(active_dates, ref_dt.date())

        return HistorySummary(
            total_sessions=int(agg["total_sessions"]),
            today_focused_sec=float(today_row["s"]),
            week_focused_sec=float(week_row["s"]),
            all_time_focused_sec=float(agg["all_focused"]),
            total_distractions=int(agg["total_distractions"]),
            best_streak_sec=float(agg["best_streak"]),
            daily_streak_days=daily_streak,
        )


def _compute_daily_streak(active_dates: set, today) -> int:
    if not active_dates:
        return 0
    cursor = today if today in active_dates else today - timedelta(days=1)
    streak = 0
    while cursor in active_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak

