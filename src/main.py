"""Air drawing with hand gestures."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import ui
from camera import CameraStream
from canvas import COLOR_NAMES, Canvas
from gestures import Gesture, GestureRecognizer
from menu import Menu
from mp_tracker import MediaPipeTracker
from renderer3d import Object3D
from smoothing import LandmarkFilter, OneEuroFilter, ScalarEMA

WINDOW = "Hand Tracking Drawing"
GRAB_HOLD_SECONDS = 0.35
OK_HOLD_SECONDS = 0.5
PINKY_HOLD_SECONDS = 0.4
WAVE_WINDOW_SECONDS = 1.3
WAVE_AMPLITUDE_RATIO = 0.35
WAVE_FLIPS = 4
WAVE_MEMORY_SECONDS = 0.7
WAVE_PALM_GRACE = 0.5
BYE_SECONDS = 1.0
STROKE_GRACE_SECONDS = 0.14
TOAST_SECONDS = 2.2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Hand tracking drawing (MediaPipe + OpenCV)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=str(root / "models" / "hand_landmarker.task"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--output", default=str(root / "output"))
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-menu", action="store_true")
    p.add_argument("--show-skeleton", action="store_true", default=True)
    p.add_argument("--video-mode", action="store_true", default=True)
    p.add_argument("--no-video-mode", dest="video_mode", action="store_false")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


class WaveDetector:
    """Side to side shaking detector."""

    def __init__(self, window: float = WAVE_WINDOW_SECONDS, flips: int = WAVE_FLIPS) -> None:
        self.window = window
        self.flips = flips
        self._dir = 0
        self._extreme: float | None = None
        self._times: deque[float] = deque()

    def reset(self) -> None:
        """Drop tracking state."""
        self._dir = 0
        self._extreme = None
        self._times.clear()

    def update(self, x: float, amplitude: float, now: float) -> bool:
        """Feed one sample."""
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()
        if self._extreme is None:
            self._extreme = x
            return False
        delta = x - self._extreme
        if abs(delta) >= amplitude:
            direction = 1 if delta > 0 else -1
            if direction != self._dir:
                if self._dir != 0:
                    self._times.append(now)
                self._dir = direction
            self._extreme = x
        return len(self._times) >= self.flips


class App:
    """Main application loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cam = CameraStream(args.camera, args.width, args.height)
        self.h, self.w = self.cam.shape

        print(f"[i] camera: {self.w}x{self.h} @ {self.cam.source_fps:.0f} FPS", flush=True)
        self.tracker = MediaPipeTracker(args.model, conf=args.conf,
                                        max_hands=args.max_hands, video_mode=args.video_mode)
        self.device = self.tracker.device
        print(f"[i] backend: mediapipe  device: {self.device}", flush=True)
        print(f"[i] model: {args.model}", flush=True)
        self.recognizers = [GestureRecognizer() for _ in range(args.max_hands)]
        self._slot_centers: list[np.ndarray | None] = [None] * args.max_hands
        self._kp_filters = [LandmarkFilter() for _ in range(args.max_hands)]
        self.canvas = Canvas(self.w, self.h, args.output)
        self.tip_filter = OneEuroFilter(min_cutoff=1.0, beta=0.06)

        self.mode_3d = False
        self.object3d: Object3D | None = None
        self._object_rev = -1
        self.yaw, self.pitch, self.roll = 0.4, -0.25, 0.0
        self.scale = 1.0
        self.yaw_s, self.pitch_s = ScalarEMA(0.4, self.yaw), ScalarEMA(0.4, self.pitch)
        self.scale_s = ScalarEMA(0.3, 1.0)
        self._grab: tuple[np.ndarray, float, float] | None = None
        self._grab_span: tuple[float, float] | None = None

        self.show_help = False
        self.show_debug = args.debug
        self.show_skeleton = args.show_skeleton
        self.grab_since: float | None = None
        self._ok_since: float | None = None
        self._ok_latched = False
        self._palm_armed = True
        self._middle_latched = False
        self._pinky_since: float | None = None
        self._pinky_latched = False
        self._wave = [WaveDetector() for _ in range(args.max_hands)]
        self._wave_until = [0.0] * args.max_hands
        self._palm_seen = [0.0] * args.max_hands
        self._bye_at: float | None = None
        self._draw_lost_since: float | None = None
        self._zoom_anchor: list[float] | None = None
        self.toast_text = ""
        self.toast_until = 0.0
        self.fps = 0.0
        self._last_t = time.perf_counter()
        self._last_render: np.ndarray | None = None

    def toast(self, text: str, seconds: float = TOAST_SECONDS) -> None:
        """Show transient message."""
        self.toast_text = text
        self.toast_until = time.time() + seconds

    def build_object(self) -> bool:
        """Rebuild 3D object."""
        if self.object3d is not None and self._object_rev == self.canvas.revision:
            return True
        crop = self.canvas.cropped_bgra(padding=28)
        if crop is None:
            self.toast("Draw something first")
            return False
        self.object3d = Object3D(crop)
        self._object_rev = self.canvas.revision
        return True

    def enter_3d(self) -> None:
        """Switch to 3D mode."""
        if self.build_object():
            self.mode_3d = True
            self._palm_armed = False
            self.canvas.end()
            self.toast("Grabbed. Move fist to rotate")

    def exit_3d(self) -> None:
        """Return to drawing mode."""
        self.mode_3d = False
        self._grab = None
        self._grab_span = None
        self.toast("Back to drawing")

    def clear_canvas(self) -> None:
        """Wipe the board."""
        self.stop_stroke(force=True)
        self.canvas.clear()
        self.object3d = None
        self._object_rev = -1
        self._grab = None
        self._grab_span = None
        self.grab_since = None
        if self.mode_3d:
            self.mode_3d = False
        self.toast("Canvas cleared")

    def handle_ok(self, states: list) -> None:
        """Clear on OK gesture."""
        ok = any(s.gesture == Gesture.OK and s.stable for s in states)
        if not ok:
            self._ok_since = None
            self._ok_latched = False
            return
        now = time.time()
        self._ok_since = self._ok_since or now
        if not self._ok_latched and now - self._ok_since >= OK_HOLD_SECONDS:
            self._ok_latched = True
            self.clear_canvas()

    def handle_middle(self, states: list) -> None:
        """React to middle finger."""
        shown = any(s.gesture == Gesture.MIDDLE and s.stable for s in states)
        if not shown:
            self._middle_latched = False
            return
        if not self._middle_latched:
            self._middle_latched = True
            self.toast(":((", 1.0)

    def handle_pinky(self, states: list) -> None:
        """Undo on pinky gesture."""
        shown = any(s.gesture == Gesture.PINKY and s.stable for s in states)
        if not shown:
            self._pinky_since = None
            self._pinky_latched = False
            return
        now = time.time()
        self._pinky_since = self._pinky_since or now
        if not self._pinky_latched and now - self._pinky_since >= PINKY_HOLD_SECONDS:
            self._pinky_latched = True
            if self.canvas.strokes:
                self.stop_stroke(force=True)
                self.canvas.undo()
                self.toast("Undo last stroke")
            else:
                self.toast("Nothing to undo")

    def handle_draw_mode(self, states: list, dt: float) -> None:
        """Drive drawing and grab."""
        drawing = [s for s in states if s.gesture == Gesture.DRAW]
        grabbing = [s for s in states if s.gesture == Gesture.GRAB]
        now = time.time()

        if len(grabbing) >= 2:
            self.stop_stroke(force=True)
            self.zoom_canvas(grabbing)
            self.grab_since = None
            return
        self._zoom_anchor = None

        if drawing:
            state = drawing[0]
            self.canvas.begin()
            self.canvas.add_point(self.tip_filter(state.cursor, dt))
            self._draw_lost_since = None
            self.grab_since = None
            return

        self.stop_stroke()

        if grabbing:
            self.grab_since = self.grab_since or now
            if now - self.grab_since >= GRAB_HOLD_SECONDS:
                self.grab_since = None
                self.enter_3d()
        else:
            self.grab_since = None

    def stop_stroke(self, force: bool = False) -> None:
        """End stroke after grace."""
        if force:
            self._draw_lost_since = None
            self.canvas.end()
            self.tip_filter.reset()
            return
        now = time.time()
        self._draw_lost_since = self._draw_lost_since or now
        if now - self._draw_lost_since >= STROKE_GRACE_SECONDS:
            self.canvas.end()
            self.tip_filter.reset()

    def zoom_canvas(self, grabbing: list) -> None:
        """Scale canvas by fists."""
        span = float(np.linalg.norm(grabbing[0].pinch_point - grabbing[1].pinch_point))
        if self._zoom_anchor is None:
            self._zoom_anchor = [span, 1.0]
            return
        span0, applied = self._zoom_anchor
        if span0 < 1e-3:
            return
        target = float(np.clip(span / span0, 0.25, 4.0))
        delta = target / applied
        if abs(delta - 1.0) < 0.02:
            return
        span_px = self.canvas.content_span()
        diag = float(np.hypot(self.w, self.h))
        if (delta > 1 and span_px > diag * 0.95) or (delta < 1 and span_px < 40):
            return
        self.canvas.scale_content(delta)
        self._zoom_anchor[1] = target
        self.toast(f"Canvas scale {target * 100:.0f}%")

    def handle_3d_mode(self, states: list, dt: float) -> None:
        """Rotate and scale object."""
        grabbing = [s for s in states if s.gesture == Gesture.GRAB]

        if grabbing:
            p = grabbing[0].pinch_point
            if self._grab is None:
                self._grab = (p.copy(), self.yaw, self.pitch)
            anchor, yaw0, pitch0 = self._grab
            dx = (p[0] - anchor[0]) / self.w
            dy = (p[1] - anchor[1]) / self.h
            self.yaw = yaw0 + dx * 2.0 * np.pi * 1.1
            self.pitch = float(np.clip(pitch0 + dy * np.pi * 1.1, -1.3, 1.3))

            if len(grabbing) >= 2:
                span = float(np.linalg.norm(grabbing[0].pinch_point - grabbing[1].pinch_point))
                if self._grab_span is None:
                    self._grab_span = (span, self.scale)
                span0, scale0 = self._grab_span
                if span0 > 1e-3:
                    self.scale = float(np.clip(scale0 * (span / span0), 0.35, 2.6))
            else:
                self._grab_span = None
        else:
            self._grab = None
            self._grab_span = None
            self.yaw += 0.25 * dt

        palm = any(s.gesture == Gesture.OPEN_PALM for s in states)
        if not palm:
            self._palm_armed = True
        elif grabbing:
            self._palm_armed = False
        elif self._palm_armed:
            self.exit_3d()

    def assign_slots(self, hands, dt: float = 1 / 30) -> list:
        """Bind hands to slots."""
        n = len(self.recognizers)
        slots: list = [None] * n
        free = list(range(n))
        for hand in hands[:n]:
            c = hand.center
            best, best_d = None, float("inf")
            for s in free:
                prev = self._slot_centers[s]
                d = float(np.linalg.norm(c - prev)) if prev is not None else 1e6
                if d < best_d:
                    best, best_d = s, d
            if best is None:
                break
            slots[best] = hand
            free.remove(best)

        result = []
        now = time.time()
        for i in range(n):
            hand = slots[i]
            if hand is None:
                self._slot_centers[i] = None
                self._kp_filters[i].reset()
                self.recognizers[i](None)
                self._wave[i].reset()
                self._wave_until[i] = 0.0
                self._palm_seen[i] = 0.0
                continue
            hand.keypoints = self._kp_filters[i](hand.keypoints, dt)
            self._slot_centers[i] = hand.center
            state = self.recognizers[i](hand)
            self.track_wave(i, hand, state, now)
            result.append((hand, state))
        return result

    def track_wave(self, slot: int, hand, state, now: float) -> None:
        """Follow one waving hand."""
        if state.gesture == Gesture.OPEN_PALM:
            self._palm_seen[slot] = now
        if now - self._palm_seen[slot] > WAVE_PALM_GRACE:
            self._wave[slot].reset()
            self._wave_until[slot] = 0.0
            return
        amplitude = hand.scale * WAVE_AMPLITUDE_RATIO
        if self._wave[slot].update(float(hand.center[0]), amplitude, now):
            self._wave_until[slot] = now + WAVE_MEMORY_SECONDS

    def handle_goodbye(self) -> None:
        """Quit on two waving hands."""
        if self._bye_at is not None:
            return
        now = time.time()
        if sum(1 for until in self._wave_until if now < until) >= 2:
            self._bye_at = now + BYE_SECONDS
            self.toast("Bye :)", BYE_SECONDS)

    @staticmethod
    def active_state(states: list):
        """Pick the acting hand."""
        if not states:
            return None
        for wanted in (Gesture.DRAW, Gesture.GRAB, Gesture.MOVE):
            for s in states:
                if s.gesture == wanted:
                    return s
        return max(states, key=lambda s: s.confidence)

    def render(self, frame: np.ndarray, hands, states) -> np.ndarray:
        """Compose the output frame."""
        if self.mode_3d and self.object3d is not None:
            out = cv2.convertScaleAbs(frame, alpha=0.32)
            out = self.object3d.render(
                out,
                yaw=self.yaw_s(self.yaw),
                pitch=self.pitch_s(self.pitch),
                roll=self.roll,
                scale=self.scale_s(self.scale),
            )
        else:
            out = self.canvas.composite_over(frame)

        if self.show_skeleton:
            for hand, st in zip(hands, states):
                ui.draw_hand(out, hand, st)
        active = self.active_state(states)
        if not self.mode_3d and active is not None:
            ui.draw_cursor(out, active.cursor, active.gesture,
                           self.canvas.color, self.canvas.thickness)

        progress = 0.0
        if self.grab_since is not None:
            progress = (time.time() - self.grab_since) / GRAB_HOLD_SECONDS

        shown = Gesture.NONE if active is None else active.gesture
        if not self.mode_3d and len([s for s in states if s.gesture == Gesture.GRAB]) >= 2:
            shown = Gesture.ZOOM
        ui.draw_hud(
            out,
            gesture=shown,
            hands_info=[(s.handedness, s.gesture) for s in states],
            mode_3d=self.mode_3d,
            fps=self.fps,
            device=self.device,
            color=self.canvas.color,
            thickness=self.canvas.thickness,
            strokes=len(self.canvas.strokes),
            clear_progress=progress,
            toast=self.toast_text if time.time() < self.toast_until else "",
        )
        if self.show_debug and states:
            ui.draw_debug(out, hands, states)
        if self.show_help:
            ui.draw_help(out)
        return out

    def handle_key(self, key: int) -> bool:
        """Handle one keypress."""
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            if self.canvas.is_empty:
                self.toast("Canvas is empty")
            else:
                paths = self.canvas.save()
                print("[+] saved:", flush=True)
                for p in paths:
                    print("   ", p, flush=True)
                self.toast(f"Saved {paths[0].name}")
        elif key == ord("c"):
            self.clear_canvas()
        elif key == ord("u"):
            self.canvas.undo()
            self.toast("Undo last stroke")
        elif key in (ord("["), ord("-"), ord(",")):
            self.canvas.set_thickness(self.canvas.thickness - 2)
            self.toast(f"Size {self.canvas.thickness}")
        elif key in (ord("]"), ord("="), ord("+"), ord(".")):
            self.canvas.set_thickness(self.canvas.thickness + 2)
            self.toast(f"Size {self.canvas.thickness}")
        elif key == ord("r"):
            self.yaw, self.pitch, self.scale = 0.4, -0.25, 1.0
            self.yaw_s.set(self.yaw)
            self.pitch_s.set(self.pitch)
            self.scale_s.set(1.0)
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("k"):
            self.show_skeleton = not self.show_skeleton
        elif key == ord("d"):
            self.show_debug = not self.show_debug
            self.toast("Debug " + ("on" if self.show_debug else "off"))
        elif key == ord(" "):
            if self.mode_3d:
                self.exit_3d()
            else:
                self.enter_3d()
        elif ord("1") <= key <= ord("6"):
            self.canvas.set_color(key - ord("1"))
            self.toast(f"Color {COLOR_NAMES[(key - ord('1')) % len(COLOR_NAMES)]}")
        return True

    def run(self) -> None:
        """Run the main loop."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.w, self.h)
        print("[i] window open. H help, Q quit.", flush=True)
        self.toast("One finger to draw. H for help")

        reason = "loop finished"
        frames = 0
        seq = -1
        while True:
            frame, seq = self.cam.wait_next(seq)
            if self.cam.failed:
                reason = "camera stopped delivering frames"
                break
            if not self.args.no_mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[:2] != (self.h, self.w):
                frame = cv2.resize(frame, (self.w, self.h))

            now = time.perf_counter()
            dt = max(now - self._last_t, 1e-4)
            self._last_t = now
            self.fps += 0.12 * (1.0 / dt - self.fps)

            detected = self.tracker(frame)
            paired = self.assign_slots(detected, dt)
            hands = [h for h, _ in paired]
            states = [s for _, s in paired]

            self.handle_ok(states)
            self.handle_middle(states)
            self.handle_pinky(states)
            self.handle_goodbye()

            if self.mode_3d:
                self.handle_3d_mode(states, dt)
            else:
                self.handle_draw_mode(states, dt)

            out = self.render(frame, hands, states)
            self._last_render = out
            cv2.imshow(WINDOW, out)

            key = cv2.waitKey(1) & 0xFF
            frames += 1
            if frames % 150 == 0:
                print(f"[i] frames: {frames}  {self.fps:.1f} FPS  hands: {len(hands)}", flush=True)

            if key != 255 and not self.handle_key(key):
                reason = f"quit key pressed ({key})"
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                reason = "window closed"
                break
            if self._bye_at is not None and time.time() >= self._bye_at:
                reason = "goodbye wave"
                break

        print(f"[i] stopped: {reason} (frames processed: {frames})", flush=True)
        self.cam.release()
        cv2.destroyAllWindows()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    if not args.no_menu and Menu().run() is None:
        print("[i] no mode selected", flush=True)
        return 0
    try:
        App(args).run()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[i] stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
