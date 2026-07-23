from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from canvas import PALETTE
from gestures import LABELS, Gesture, GestureState
from hand import HAND_CONNECTIONS, INDEX_TIP, Hand

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_ACCENT = {
    Gesture.DRAW: (90, 230, 90),
    Gesture.MOVE: (255, 190, 60),
    Gesture.GRAB: (60, 220, 255),
    Gesture.OPEN_PALM: (230, 230, 230),
    Gesture.ZOOM: (255, 110, 200),
    Gesture.OK: (120, 120, 255),
    Gesture.MIDDLE: (80, 80, 240),
    Gesture.PINKY: (200, 160, 255),
    Gesture.NONE: (150, 150, 150),
}

HELP_LINES = (
    "GESTURES",
    "  One index finger - draw",
    "  Index and middle - move cursor",
    "  Fist (hold) - grab, move to rotate in 3D",
    "  Two fists - scale",
    "  Open palm - release",
    "  OK (hold) - clear canvas",
    "  Pinky (hold) - undo last stroke",
    "  Middle finger - :((",
    "  Two palms waving - quit",
    "",
    "KEYS",
    "  S save    C clear    U undo",
    "  1..6 color    [ ] size    R reset view",
    "  SPACE 2D/3D    K skeleton    D debug",
    "  H help    Q quit",
)


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    """Find available font."""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font at size."""
    path = _font_path()
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=256)
def _render_text(text: str, size: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize and cache text."""
    font = _font(size)
    dummy = ImageDraw.Draw(Image.new("L", (1, 1)))
    box = dummy.textbbox((0, 0), text, font=font)
    w = max(1, box[2] - box[0] + 4)
    h = max(1, box[3] - box[1] + 4)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=font, fill=255)
    alpha = np.array(img, dtype=np.float32) / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :] = color
    return rgb, alpha


def put_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 18,
    color: tuple[int, int, int] = (240, 240, 240),
) -> int:
    """Draw text string."""
    if not text:
        return size + 6
    rgb, alpha = _render_text(text, size, color)
    h, w = alpha.shape
    x, y = org
    fh, fw = frame.shape[:2]
    if x >= fw or y >= fh:
        return h
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return h
    sub_a = alpha[y0 - y : y1 - y, x0 - x : x1 - x][:, :, None]
    sub_c = rgb[y0 - y : y1 - y, x0 - x : x1 - x]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - sub_a) + sub_c * sub_a).astype(np.uint8)
    return h


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Dark translucent backing."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)


def draw_hand(frame: np.ndarray, hand: Hand, state: GestureState, kpt_conf: float = 0.3) -> None:
    """Draw hand skeleton."""
    kp = hand.keypoints
    ok = hand.scores >= kpt_conf
    color = _ACCENT.get(state.gesture, (150, 150, 150))
    for a, b in HAND_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (60, 60, 60), 3, cv2.LINE_AA)
            cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     color, 1, cv2.LINE_AA)
    for i in range(21):
        if ok[i]:
            cv2.circle(frame, tuple(kp[i].astype(int)), 3, color, -1, cv2.LINE_AA)
    if state.gesture == Gesture.DRAW:
        cv2.circle(frame, tuple(kp[INDEX_TIP].astype(int)), 9, (90, 230, 90), 2, cv2.LINE_AA)
    elif state.gesture == Gesture.GRAB:
        cv2.circle(frame, tuple(hand.center.astype(int)), 16, (60, 220, 255), 2, cv2.LINE_AA)


def draw_cursor(frame: np.ndarray, point: np.ndarray, gesture: Gesture,
                color: tuple[int, int, int], thickness: int) -> None:
    """Draw pen cursor."""
    p = tuple(np.asarray(point, dtype=int))
    if gesture == Gesture.DRAW:
        cv2.circle(frame, p, max(4, thickness), color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, max(4, thickness) + 5, (255, 255, 255), 1, cv2.LINE_AA)
    elif gesture == Gesture.MOVE:
        cv2.circle(frame, p, max(6, thickness + 4), (255, 190, 60), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, p, (255, 190, 60), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    *,
    gesture: Gesture,
    hands_info: list | None = None,
    mode_3d: bool,
    fps: float,
    device: str,
    color: tuple[int, int, int],
    thickness: int,
    strokes: int,
    clear_progress: float = 0.0,
    toast: str = "",
) -> None:
    """Draw status overlay."""
    h, w = frame.shape[:2]

    panel(frame, 0, 0, w, 78, 0.5)
    mode = "3D VIEW" if mode_3d else "DRAW"
    put_text(frame, mode, (16, 10), 20, (255, 255, 255))
    put_text(frame, f"Gesture: {LABELS[gesture]}", (16, 40), 17, _ACCENT.get(gesture, (200, 200, 200)))

    hands_txt = ", ".join(f"{h}" for h, _ in (hands_info or [])) or "no hands"
    right = f"{fps:.0f} FPS   {device.upper()}   strokes {strokes}   {hands_txt}"
    rw = _render_text(right, 16, (200, 200, 200))[0].shape[1]
    put_text(frame, right, (w - rw - 16, 14), 16, (200, 200, 200))
    put_text(frame, "H help", (w - rw - 16, 40), 15, (140, 140, 140))

    panel(frame, 0, h - 46, w, 46, 0.5)
    x = 16
    for c in PALETTE:
        cv2.rectangle(frame, (x, h - 34), (x + 24, h - 12), c, -1, cv2.LINE_AA)
        if c == color:
            cv2.rectangle(frame, (x - 3, h - 37), (x + 27, h - 9), (255, 255, 255), 2, cv2.LINE_AA)
        x += 34
    put_text(frame, f"size {thickness}", (x + 10, h - 34), 16, (220, 220, 220))
    hint = "Finger draw    Fist 3D    Palm release"
    hw = _render_text(hint, 15, (170, 170, 170))[0].shape[1]
    put_text(frame, hint, (w - hw - 16, h - 32), 15, (170, 170, 170))
    cv2.circle(frame, (x + 130 + thickness, h - 23), max(1, thickness // 2), color, -1, cv2.LINE_AA)

    if clear_progress > 0:
        bw = int(260 * min(1.0, clear_progress))
        cv2.rectangle(frame, (w // 2 - 130, h - 78), (w // 2 + 130, h - 62), (40, 40, 40), -1)
        cv2.rectangle(frame, (w // 2 - 130, h - 78), (w // 2 - 130 + bw, h - 62), (80, 80, 255), -1)
        put_text(frame, "grabbing", (w // 2 - 30, h - 104), 16, (200, 240, 255))

    if toast:
        tw = _render_text(toast, 18, (255, 255, 255))[0].shape[1]
        panel(frame, w // 2 - tw // 2 - 14, 92, tw + 28, 36, 0.65)
        put_text(frame, toast, (w // 2 - tw // 2, 100), 18, (255, 255, 255))


def draw_debug(frame: np.ndarray, hands, states) -> None:
    """Draw classifier debug panel."""
    names = ("thumb", "index", "middl", "ring ", "pinky")
    y = 96
    for hand, st in zip(hands, states):
        panel(frame, 12, y, 300, 24 + 16 * 7, 0.66)
        yy = y + 16
        cv2.putText(frame, f"{st.handedness[:4]} conf {st.confidence:.2f} "
                           f"{'stable' if st.stable else 'JITTER'}",
                    (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        yy += 16
        for i, name in enumerate(names):
            ratio = st.ratios[i] if i < len(st.ratios) else 1.0
            curl = st.curls[i] if i < len(st.curls) else 0.0
            up = st.fingers[i]
            col = (110, 240, 110) if up else (140, 140, 220)
            bar = "#" * int(min(max(ratio - 0.7, 0) * 12, 10))
            cv2.putText(frame, f"{name} x{ratio:4.2f} {curl:3.0f}d "
                               f"{'UP  ' if up else 'down'} {bar}",
                        (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            yy += 16
        closed = st.pinch_distance < 0.62
        cv2.putText(frame, f"pinch {st.pinch_distance:5.2f} "
                           f"{'CLOSED' if closed else 'open'}  lowconf "
                           f"{int((hand.scores < 0.25).sum())}/21",
                    (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (110, 240, 110) if closed else (180, 180, 180), 1, cv2.LINE_AA)
        y += 24 + 16 * 7 + 8


def draw_help(frame: np.ndarray) -> None:
    """Draw help overlay."""
    h, w = frame.shape[:2]
    bw, bh = min(w - 60, 720), 20 + len(HELP_LINES) * 26
    x, y = (w - bw) // 2, (h - bh) // 2
    panel(frame, x, y, bw, bh, 0.78)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (90, 90, 90), 1, cv2.LINE_AA)
    cy = y + 12
    for line in HELP_LINES:
        bold = line and not line.startswith(" ")
        put_text(frame, line, (x + 22, cy), 17 if bold else 16,
                 (255, 255, 255) if bold else (215, 215, 215))
        cy += 26
