"""System tray icon & menu controller for background / headless operation."""

from __future__ import annotations

import threading
from typing import Callable

from PIL import Image, ImageDraw

from .state import FocusState, SessionSnapshot


_STATE_COLORS: dict[FocusState, tuple[int, int, int]] = {
    FocusState.FOCUSED: (34, 197, 94),
    FocusState.GRACE: (245, 158, 11),
    FocusState.DISTRACTED: (239, 68, 68),
    FocusState.AWAY: (59, 130, 246),
    FocusState.PAUSED: (148, 163, 184),
    FocusState.BREAK: (6, 182, 212),
}


def create_status_icon(state: FocusState, size: int = 64) -> Image.Image:
    """Render a crisp circular shield/status tray icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = _STATE_COLORS.get(state, (34, 197, 94))
    pad = max(4, size // 12)
    draw.ellipse((pad, pad, size - pad, size - pad), fill=color, outline=(255, 255, 255, 220), width=max(2, size // 16))
    inner = size // 3
    draw.ellipse((inner, inner, size - inner, size - inner), fill=(15, 23, 42, 220))
    return img


class TrayController:
    """Runs a pystray system tray icon in a background thread."""

    def __init__(
        self,
        on_toggle_preview: Callable[[], None],
        on_toggle_pause: Callable[[], None],
        on_toggle_break: Callable[[], None],
        on_toggle_sound: Callable[[], None],
        on_reset: Callable[[], None],
        on_quit: Callable[[], None],
        is_preview_visible: Callable[[], bool],
        is_sound_enabled: Callable[[], bool],
    ) -> None:
        self._on_toggle_preview = on_toggle_preview
        self._on_toggle_pause = on_toggle_pause
        self._on_toggle_break = on_toggle_break
        self._on_toggle_sound = on_toggle_sound
        self._on_reset = on_reset
        self._on_quit = on_quit
        self._is_preview_visible = is_preview_visible
        self._is_sound_enabled = is_sound_enabled

        self._icon = None
        self._thread: threading.Thread | None = None
        self._last_state: FocusState = FocusState.FOCUSED
        self._status_line: str = "Focused | Distractions: 0"

    def start(self) -> bool:
        """Start the tray icon thread. Returns False if tray backend is unavailable."""
        try:
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem(lambda _item: self._status_line, None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    lambda _item: "Hide Preview" if self._is_preview_visible() else "Show Preview",
                    lambda _icon, _item: self._on_toggle_preview(),
                ),
                pystray.MenuItem(
                    lambda _item: "Resume Focus" if self._last_state == FocusState.PAUSED else "Pause Session",
                    lambda _icon, _item: self._on_toggle_pause(),
                ),
                pystray.MenuItem(
                    lambda _item: "End Break" if self._last_state == FocusState.BREAK else "Start Break",
                    lambda _icon, _item: self._on_toggle_break(),
                ),
                pystray.MenuItem(
                    lambda _item: "Mute Sound" if self._is_sound_enabled() else "Enable Sound",
                    lambda _icon, _item: self._on_toggle_sound(),
                ),
                pystray.MenuItem("Reset Session", lambda _icon, _item: self._on_reset()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit & Save", lambda _icon, _item: self._on_quit()),
            )
            self._icon = pystray.Icon(
                "focus-defence",
                create_status_icon(FocusState.FOCUSED),
                "Focus-Defence",
                menu=menu,
            )
            self._thread = threading.Thread(target=self._icon.run, name="focus-tray", daemon=True)
            self._thread.start()
            return True
        except Exception:
            self._icon = None
            return False

    def update(self, snapshot: SessionSnapshot) -> None:
        """Refresh tray tooltip, status label, and icon color on state change."""
        mins, secs = divmod(int(snapshot.focused_sec), 60)
        self._status_line = (
            f"{snapshot.state.value.title()} | Focus: {mins:02d}:{secs:02d} | "
            f"Distractions: {snapshot.distraction_count}"
        )
        if self._icon is None:
            return
        self._icon.title = f"Focus-Defence: {self._status_line}"
        if snapshot.state != self._last_state:
            self._last_state = snapshot.state
            try:
                self._icon.icon = create_status_icon(snapshot.state)
                self._icon.update_menu()
            except Exception:
                pass

    def notify(self, message: str, title: str = "Focus-Defence") -> None:
        """Show a native OS notification balloon via the tray icon."""
        if self._icon is None:
            return
        try:
            self._icon.notify(message, title)
        except Exception:
            pass

    def stop(self) -> None:
        """Remove the tray icon cleanly."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None

