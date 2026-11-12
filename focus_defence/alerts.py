"""Non-blocking cross-platform audio alarms and desktop notification alerts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

from .state import FocusState, SessionSnapshot


class AlertManager:
    """Dispatches non-blocking audio beeps and desktop notifications with cooldowns."""

    def __init__(
        self,
        sound_enabled: bool = False,
        notifications_enabled: bool = True,
        notify_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.sound_enabled = sound_enabled
        self.notifications_enabled = notifications_enabled
        self._notify_callback = notify_callback

        self._audio_lock = threading.Lock()
        self._last_beep_time = 0.0
        self._last_notify_time = 0.0

    def set_notify_callback(self, callback: Callable[[str, str], None] | None) -> None:
        """Attach or update the desktop notification callback (e.g. system tray)."""
        self._notify_callback = callback

    def toggle_sound(self) -> bool:
        """Toggle sound alerts on/off and return the new state."""
        self.sound_enabled = not self.sound_enabled
        return self.sound_enabled

    def process_snapshot(self, snapshot: SessionSnapshot, now: float | None = None) -> None:
        """React to state-machine events and ongoing distraction alert levels."""
        t = time.monotonic() if now is None else now

        # One-shot event notifications
        for event in snapshot.events:
            if event == "entered_distracted":
                self._send_notification("Focus-Defence: Distraction!", "Phone detected — put it away to protect your streak.")
                if self.sound_enabled:
                    self._play_tone_async([(1100, 180)], t, min_cooldown=0.4)
            elif event == "escalated":
                self._send_notification("Focus-Defence: Still on Phone!", "Distraction escalated — get back into deep focus.")
                if self.sound_enabled:
                    self._play_tone_async([(1400, 180), (1400, 180)], t, min_cooldown=0.4)
            elif event == "break_started":
                self._send_notification("Focus-Defence: Break Time!", "Work block complete. Rest your eyes and recharge.")
                if self.sound_enabled:
                    self._play_tone_async([(784, 140), (1046, 200)], t, min_cooldown=0.4)
            elif event == "break_ended":
                self._send_notification("Focus-Defence: Break Over", "Time to lock back in for the next focus block.")
                if self.sound_enabled:
                    self._play_tone_async([(1046, 140), (784, 180)], t, min_cooldown=0.4)

        # Periodic reminder beep while actively distracted
        if self.sound_enabled and snapshot.state == FocusState.DISTRACTED:
            if snapshot.alert_level >= 3:
                self._play_tone_async([(1450, 220)], t, min_cooldown=1.2)
            else:
                self._play_tone_async([(1150, 180)], t, min_cooldown=2.0)

    def _send_notification(self, title: str, message: str) -> None:
        if not self.notifications_enabled or self._notify_callback is None:
            return
        now = time.monotonic()
        if (now - self._last_notify_time) < 2.0:
            return
        self._last_notify_time = now
        try:
            self._notify_callback(message, title)
        except Exception:
            # Never crash the main loop if OS notification backend fails
            pass

    def _play_tone_async(self, notes: list[tuple[int, int]], now: float, min_cooldown: float) -> None:
        if (now - self._last_beep_time) < min_cooldown:
            return
        if not self._audio_lock.acquire(blocking=False):
            return
        self._last_beep_time = now
        thread = threading.Thread(target=self._run_notes, args=(notes,), daemon=True)
        thread.start()

    def _run_notes(self, notes: list[tuple[int, int]]) -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound

                for freq, duration_ms in notes:
                    winsound.Beep(freq, duration_ms)
                    time.sleep(0.04)
            else:
                self._play_posix_fallback()
        except Exception:
            pass
        finally:
            self._audio_lock.release()

    @staticmethod
    def _play_posix_fallback() -> None:
        if sys.platform == "darwin" and shutil.which("afplay"):
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif shutil.which("paplay"):
            subprocess.Popen(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

