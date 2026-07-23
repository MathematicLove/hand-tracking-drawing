from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from hand import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    Hand,
)


class Gesture(str, Enum):
    NONE = "none"
    DRAW = "draw"
    MOVE = "move"
    GRAB = "grab"
    OPEN_PALM = "palm"
    ZOOM = "zoom"
    OK = "ok"
    MIDDLE = "middle"
    PINKY = "pinky"


LABELS: dict[Gesture, str] = {
    Gesture.NONE: "none",
    Gesture.DRAW: "Draw",
    Gesture.MOVE: "Move",
    Gesture.GRAB: "Grab",
    Gesture.OPEN_PALM: "Palm",
    Gesture.ZOOM: "Zoom",
    Gesture.OK: "OK",
    Gesture.MIDDLE: "Middle",
    Gesture.PINKY: "Undo",
}

_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

_E_LO = 0.96
_FIST_MAX = 1.04
_UP_ON = 0.72
_UP_OFF = 0.60
_UP_LOW_ON = 0.82
_UP_LOW_OFF = 0.50

_PINCH_ON = 0.42
_PINCH_OFF = 0.62

_MIN_KPT_CONF = 0.25
_STRAIGHT_MARGIN = 32.0


@dataclass
class GestureState:
    """One hand classification result."""

    gesture: Gesture = Gesture.NONE
    fingers: tuple[bool, bool, bool, bool, bool] = (False,) * 5
    curls: tuple[float, ...] = (0.0,) * 5
    ratios: tuple[float, ...] = (1.0,) * 5
    pinch_distance: float = 1.0
    cursor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    confidence: float = 0.0
    handedness: str = "?"
    stable: bool = False


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def finger_curl(kp: np.ndarray, chain: tuple[int, int, int, int]) -> float:
    """Finger bend in degrees."""
    mcp, pip, _dip, tip = chain
    return _angle(kp[pip] - kp[mcp], kp[tip] - kp[pip])


def handedness(kp: np.ndarray) -> str:
    """Left or right hand."""
    v1 = kp[INDEX_MCP] - kp[WRIST]
    v2 = kp[PINKY_MCP] - kp[WRIST]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return "right" if z > 0 else "left"


class GestureRecognizer:
    """Single hand gesture classifier."""

    def __init__(self, vote_window: int = 5, vote_ratio: float = 0.6) -> None:
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio
        self._history: deque[Gesture] = deque(maxlen=vote_window)
        self._stable = Gesture.NONE
        self._extended = [False] * 5
        self._extendedness = [0.0] * 5
        self._fist = False
        self._seen = False
        self._pinching = False

    def reset(self) -> None:
        """Clear tracking state."""
        self._history.clear()
        self._stable = Gesture.NONE
        self._extended = [False] * 5
        self._extendedness = [0.0] * 5
        self._fist = False
        self._seen = False
        self._pinching = False

    def _update_fingers(self, kp: np.ndarray, scores: np.ndarray) -> tuple[list[float], list[float]]:
        """Decide each finger up."""
        curls, ratios = [], []
        for i, chain in enumerate(_CHAINS):
            mcp, pip, _dip, tip = chain
            curls.append(finger_curl(kp, chain))
            if i == 0:
                ref, near = kp[PINKY_MCP], kp[mcp]
            else:
                ref, near = kp[WRIST], kp[pip]
            d_far = float(np.linalg.norm(kp[tip] - ref))
            d_near = float(np.linalg.norm(near - ref))
            ratios.append(d_far / max(d_near, 1e-6))

        hi = max(ratios[1:])
        self._fist = hi < _FIST_MAX
        span = max(hi - _E_LO, 1e-3)

        for i, chain in enumerate(_CHAINS):
            mcp, pip, _dip, tip = chain
            e = float(np.clip((ratios[i] - _E_LO) / span, 0.0, 1.0))
            self._extendedness[i] = e
            sure = min(float(scores[tip]), float(scores[mcp if i == 0 else pip]))

            if not self._seen:
                self._extended[i] = e > _UP_ON
                continue
            on, off = (_UP_LOW_ON, _UP_LOW_OFF) if sure < _MIN_KPT_CONF else (_UP_ON, _UP_OFF)
            self._extended[i] = e > (off if self._extended[i] else on)
        self._seen = True
        return curls, ratios

    def __call__(self, hand: Hand | None) -> GestureState:
        """Classify one hand."""
        if hand is None:
            self.reset()
            return GestureState()

        kp, scores = hand.keypoints, hand.scores
        curls, ratios = self._update_fingers(kp, scores)

        finger_len = float(np.linalg.norm(kp[INDEX_TIP] - kp[INDEX_PIP])) + \
            float(np.linalg.norm(kp[INDEX_PIP] - kp[INDEX_MCP]))
        palm_w = float(np.linalg.norm(kp[PINKY_MCP] - kp[INDEX_MCP]))
        ref = max(finger_len, palm_w * 0.85, 1e-3)
        pinch_d = float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]) / ref)
        self._pinching = pinch_d < (_PINCH_OFF if self._pinching else _PINCH_ON)

        raw = self._classify(tuple(self._extended), tuple(self._extendedness),
                             tuple(curls), self._fist, self._pinching)

        self._history.append(raw)
        counts = Counter(self._history)
        winner, n = counts.most_common(1)[0]
        stable = n >= max(2, int(self.vote_window * self.vote_ratio))
        if stable:
            self._stable = winner

        pinch_point = (kp[THUMB_TIP] + kp[INDEX_TIP]) * 0.5
        return GestureState(
            gesture=self._stable,
            fingers=tuple(self._extended),
            curls=tuple(curls),
            ratios=tuple(ratios),
            pinch_distance=pinch_d,
            cursor=kp[INDEX_TIP].copy(),
            pinch_point=pinch_point,
            confidence=hand.conf,
            handedness=handedness(kp),
            stable=stable,
        )

    @staticmethod
    def _classify(ext: tuple[bool, ...], e: tuple[float, ...],
                  curls: tuple[float, ...], fist: bool, pinching: bool) -> Gesture:
        """Map fingers to gesture."""
        _thumb, index, middle, ring, pinky = ext
        long_up = sum((index, middle, ring, pinky))

        if pinching and middle and ring and pinky and not index:
            return Gesture.OK

        if middle and not index and not ring and not pinky:
            return Gesture.MIDDLE

        if pinky and not index and not middle and not ring:
            return Gesture.PINKY

        if fist:
            return Gesture.GRAB

        if index and middle:
            base = min(curls[1], curls[2])
            if (curls[3] < base + _STRAIGHT_MARGIN
                    and curls[4] < base + _STRAIGHT_MARGIN):
                return Gesture.OPEN_PALM
            return Gesture.MOVE

        if index:
            return Gesture.DRAW
        if long_up <= 1:
            return Gesture.GRAB
        return Gesture.NONE
