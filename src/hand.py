from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Hand:
    """One detected hand."""

    keypoints: np.ndarray
    scores: np.ndarray
    box: np.ndarray
    conf: float

    @property
    def scale(self) -> float:
        """Characteristic hand size."""
        palm = np.linalg.norm(self.keypoints[MIDDLE_MCP] - self.keypoints[WRIST])
        if palm < 1e-3:
            palm = max(self.box[2] - self.box[0], self.box[3] - self.box[1]) * 0.5
        return float(max(palm, 1e-3))

    @property
    def center(self) -> np.ndarray:
        """Palm center point."""
        return self.keypoints[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)

    def point(self, idx: int) -> np.ndarray:
        """Keypoint by index."""
        return self.keypoints[idx]
