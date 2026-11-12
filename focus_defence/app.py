"""Main application orchestration for Focus-Defence."""

from __future__ import annotations

import queue
import sys

import cv2

from .alerts import AlertManager
from .config import FocusConfig, load_config
from .detector import CameraStream, PhoneDetector, export_model_to_onnx
from .state import FocusStateMachine
from .storage import SessionStore
from .tray import TrayController
from .ui import WINDOW_NAME, format_duration, format_history_report, render_hud


class FocusDefenceApp:
    """Coordinates camera capture, AI detection, state machine, alerts, tray, and storage."""

    def __init__(self, config: FocusConfig) -> None:
        self.config = config
        self.store = SessionStore(config.db_path)
        self.state_machine = FocusStateMachine(config)
        self.alerts = AlertManager(
            sound_enabled=config.sound_enabled,
            notifications_enabled=config.notifications_enabled,
        )
        self.preview_visible = not config.headless
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._running = True
        self._window_open = False
        self._tray: TrayController | None = None

    def run(self) -> int:
        """Start camera, AI worker, and optional tray, then run the main event loop."""
        camera = CameraStream(self.config.camera_index)
        if not camera.is_opened():
            print(
                f"[Focus-Defence] Error: Could not open camera index {self.config.camera_index}.",
                file=sys.stderr,
            )
            return 1

        camera.start()
        detector = PhoneDetector(self.config)
        detector.start()

        if self.config.tray_enabled:
            self._tray = TrayController(
                on_toggle_preview=lambda: self._commands.put("toggle_preview"),
                on_toggle_pause=lambda: self._commands.put("toggle_pause"),
                on_toggle_break=lambda: self._commands.put("toggle_break"),
                on_toggle_sound=lambda: self._commands.put("toggle_sound"),
                on_reset=lambda: self._commands.put("reset"),
                on_quit=lambda: self._commands.put("quit"),
                is_preview_visible=lambda: self.preview_visible,
                is_sound_enabled=lambda: self.alerts.sound_enabled,
            )
            if self._tray.start():
                self.alerts.set_notify_callback(self._tray.notify)

        print(f"[Focus-Defence] Running (model={self.config.model_path}, imgsz={self.config.imgsz}, camera={self.config.camera_index}).")
        print("Privacy: 100% local processing — no frames are saved or uploaded.")
        print("Controls: 'p'=pause | 'b'=break | 'v'=hide/show | 'm'=sound | '['/']'=sens | 'r'=reset | 'q'=quit")

        try:
            while self._running and camera.is_opened():
                self._drain_commands(detector)
                if not self._running:
                    break

                frame, cam_fps = camera.read_latest(timeout=0.05)
                if frame is None:
                    if self.preview_visible and self._window_open:
                        key = cv2.waitKey(1) & 0xFF
                        self._handle_key(key, detector)
                    continue

                detector.submit_frame(frame)
                detections = detector.get_latest_result()

                snapshot = self.state_machine.update(
                    phone_detected=detections.active_phone_detected,
                    person_detected=detections.person_detected,
                )
                self.alerts.process_snapshot(snapshot)
                if self._tray is not None:
                    self._tray.update(snapshot)

                if self.preview_visible:
                    hud = render_hud(
                        frame=frame,
                        detections=detections,
                        snapshot=snapshot,
                        config=self.config,
                        sound_enabled=self.alerts.sound_enabled,
                        cam_fps=cam_fps,
                    )
                    cv2.imshow(WINDOW_NAME, hud)
                    self._window_open = True
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key, detector)

                    # Exit if the user clicked the window close button ('X')
                    if self._window_open and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        self._running = False
                else:
                    if self._window_open:
                        cv2.destroyWindow(WINDOW_NAME)
                        self._window_open = False
        except KeyboardInterrupt:
            print("\n[Focus-Defence] Interrupted — saving session...")
        finally:
            self._shutdown(detector, camera)

        return 0

    def _drain_commands(self, detector: PhoneDetector) -> None:
        while not self._commands.empty():
            cmd = self._commands.get_nowait()
            if cmd == "quit":
                self._running = False
            elif cmd == "toggle_preview":
                self.preview_visible = not self.preview_visible
            elif cmd == "toggle_pause":
                new_state = self.state_machine.toggle_pause()
                print(f"[Focus-Defence] State -> {new_state.value}")
            elif cmd == "toggle_break":
                new_state = self.state_machine.toggle_break()
                print(f"[Focus-Defence] State -> {new_state.value}")
            elif cmd == "toggle_sound":
                enabled = self.alerts.toggle_sound()
                print(f"[Focus-Defence] Sound alerts: {'ON' if enabled else 'OFF'}")
            elif cmd == "reset":
                self._save_current_session()
                self.state_machine.reset()
                print("[Focus-Defence] Session reset.")
            elif cmd == "sens_up":
                thresh = self.config.adjust_sensitivity(-0.05)
                detector.update_confidence(thresh)
                print(f"[Focus-Defence] Sensitivity adjusted: Conf threshold = {thresh:.2f}")
            elif cmd == "sens_down":
                thresh = self.config.adjust_sensitivity(0.05)
                detector.update_confidence(thresh)
                print(f"[Focus-Defence] Sensitivity adjusted: Conf threshold = {thresh:.2f}")

    def _handle_key(self, key: int, detector: PhoneDetector) -> None:
        if key == 255:
            return
        if key == ord("q"):
            self._running = False
        elif key == ord("p"):
            self._commands.put("toggle_pause")
        elif key == ord("b"):
            self._commands.put("toggle_break")
        elif key == ord("v"):
            self._commands.put("toggle_preview")
        elif key == ord("m"):
            self._commands.put("toggle_sound")
        elif key == ord("r"):
            self._commands.put("reset")
        elif key == ord("["):
            self._commands.put("sens_up")
        elif key == ord("]"):
            self._commands.put("sens_down")
        self._drain_commands(detector)

    def _save_current_session(self) -> int | None:
        self.state_machine.finalize()
        snap = self.state_machine.snapshot()
        return self.store.save_session(
            started_at=self.state_machine.session_started_wall,
            snapshot=snap,
            incidents=self.state_machine.completed_incidents,
            min_save_sec=self.config.min_save_sec,
        )

    def _shutdown(self, detector: PhoneDetector, camera: CameraStream) -> None:
        session_id = self._save_current_session()
        snap = self.state_machine.snapshot()

        detector.stop()
        if self._tray is not None:
            self._tray.stop()
        camera.stop()
        cv2.destroyAllWindows()

        print("\n=== Session Summary ===")
        print(f"Focus Time    : {format_duration(snap.focused_sec)}")
        print(f"Distracted    : {format_duration(snap.distracted_sec)} ({snap.distraction_count} incident(s))")
        print(f"Away / Break  : {format_duration(snap.away_sec)} / {format_duration(snap.break_sec)}")
        print(f"Best Streak   : {format_duration(snap.best_streak_sec)}")
        if session_id is not None:
            print(f"Saved Session : #{session_id} -> {self.config.db_path}")
        else:
            print(f"Session < {self.config.min_save_sec:.0f}s (not saved to history).")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Focus-Defence."""
    config, args = load_config(argv)

    if args.stats:
        store = SessionStore(config.db_path)
        summary = store.get_summary()
        recent = store.get_recent_sessions(limit=10)
        print(format_history_report(summary, recent))
        return 0

    if args.export_onnx:
        print(f"[Focus-Defence] Exporting {config.model_path} to ONNX (imgsz={config.imgsz})...")
        out_path = export_model_to_onnx(config.model_path, imgsz=config.imgsz)
        print(f"[Focus-Defence] Exported ONNX model to: {out_path}")
        return 0

    app = FocusDefenceApp(config)
    return app.run()

