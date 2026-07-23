from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PALETTE: tuple[tuple[int, int, int], ...] = (
    (60, 60, 255),
    (60, 220, 255),
    (90, 230, 90),
    (255, 190, 60),
    (255, 110, 200),
    (255, 255, 255),
)

COLOR_NAMES: tuple[str, ...] = (
    "red", "yellow", "green", "blue", "pink", "white",
)


@dataclass
class Stroke:
    """One drawn stroke."""

    color: tuple[int, int, int]
    thickness: int
    points: list[tuple[int, int]] = field(default_factory=list)


class Canvas:
    """Vector drawing canvas."""

    MAX_JUMP = 0.12

    def __init__(self, width: int, height: int, output_dir: str | Path = "output") -> None:
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.strokes: list[Stroke] = []
        self._active: Stroke | None = None
        self.color: tuple[int, int, int] = PALETTE[0]
        self.thickness: int = 6
        self._layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._mask = np.zeros((height, width), dtype=np.uint8)
        self._bounds: list[int] | None = None
        self.revision = 0

    def begin(self) -> None:
        """Start a new stroke."""
        if self._active is None:
            self._active = Stroke(color=self.color, thickness=self.thickness)
            self.strokes.append(self._active)

    def add_point(self, point) -> None:
        """Append point to stroke."""
        if self._active is None:
            self.begin()
        assert self._active is not None
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        pts = self._active.points
        if pts and pts[-1] == (x, y):
            return
        if pts:
            jump = np.hypot(x - pts[-1][0], y - pts[-1][1])
            if jump > self.MAX_JUMP * np.hypot(self.width, self.height):
                self.end()
                self.begin()
                pts = self._active.points
        pts.append((x, y))
        if len(pts) >= 2:
            self._draw_segment(pts[-2], pts[-1], self._active)
        else:
            self._draw_dot((x, y), self._active)

    def end(self) -> None:
        """Finish current stroke."""
        if self._active is not None and len(self._active.points) < 2:
            if len(self._active.points) == 0:
                self.strokes.remove(self._active)
                self._rebuild()
        self._active = None

    def _grow_bounds(self, points, radius: int) -> None:
        """Extend content bounding box."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs) - radius, min(ys) - radius
        x1, y1 = max(xs) + radius, max(ys) + radius
        if self._bounds is None:
            self._bounds = [x0, y0, x1, y1]
        else:
            b = self._bounds
            b[0], b[1] = min(b[0], x0), min(b[1], y0)
            b[2], b[3] = max(b[2], x1), max(b[3], y1)

    def _draw_segment(self, a, b, stroke: Stroke) -> None:
        """Draw one line segment."""
        cv2.line(self._layer, a, b, stroke.color, stroke.thickness, cv2.LINE_AA)
        cv2.line(self._mask, a, b, 255, stroke.thickness, cv2.LINE_AA)
        self._grow_bounds((a, b), stroke.thickness // 2 + 2)
        self.revision += 1

    def _draw_dot(self, p, stroke: Stroke) -> None:
        """Draw one dot."""
        r = max(1, stroke.thickness // 2)
        cv2.circle(self._layer, p, r, stroke.color, -1, cv2.LINE_AA)
        cv2.circle(self._mask, p, r, 255, -1, cv2.LINE_AA)
        self._grow_bounds((p,), r + 2)
        self.revision += 1

    def _rebuild(self) -> None:
        """Redraw all strokes."""
        self._layer[:] = 0
        self._mask[:] = 0
        self._bounds = None
        self.revision += 1
        for s in self.strokes:
            if len(s.points) == 1:
                self._draw_dot(s.points[0], s)
            for a, b in zip(s.points, s.points[1:]):
                self._draw_segment(a, b, s)

    def undo(self) -> None:
        """Remove last stroke."""
        if self.strokes:
            self.strokes.pop()
            self._active = None
            self._rebuild()

    def clear(self) -> None:
        """Erase whole canvas."""
        self.strokes.clear()
        self._active = None
        self._rebuild()

    def scale_content(self, factor: float) -> None:
        """Scale drawing about center."""
        if not self.strokes or abs(factor - 1.0) < 1e-3:
            return
        cx, cy = self.width * 0.5, self.height * 0.5
        for s in self.strokes:
            s.points = [
                (
                    int(round(cx + (x - cx) * factor)),
                    int(round(cy + (y - cy) * factor)),
                )
                for x, y in s.points
            ]
            s.thickness = int(max(1, min(48, round(s.thickness * factor))))
        self._active = None
        self._rebuild()

    def content_span(self) -> float:
        """Drawing diagonal length."""
        box = self.content_bbox(padding=0)
        if box is None:
            return 0.0
        x0, y0, x1, y1 = box
        return float(np.hypot(x1 - x0, y1 - y0))

    def set_color(self, index: int) -> None:
        """Select palette color."""
        self.color = PALETTE[index % len(PALETTE)]

    def set_thickness(self, value: int) -> None:
        """Set stroke thickness."""
        self.thickness = int(max(1, min(48, value)))

    @property
    def is_empty(self) -> bool:
        """No strokes drawn."""
        return not any(s.points for s in self.strokes)

    @property
    def layer(self) -> np.ndarray:
        """Color drawing layer."""
        return self._layer

    @property
    def mask(self) -> np.ndarray:
        """Drawing alpha mask."""
        return self._mask

    def composite_over(self, frame: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        """Blend drawing onto frame."""
        out = frame.copy()
        box = self.content_bbox(padding=2)
        if box is None:
            return out
        x0, y0, x1, y1 = box
        roi = out[y0:y1, x0:x1]
        layer = self._layer[y0:y1, x0:x1]
        mask = self._mask[y0:y1, x0:x1]
        if opacity < 1.0:
            mask = (mask.astype(np.float32) * opacity).astype(np.uint8)
        a = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(a)
        blended = cv2.add(
            cv2.multiply(roi, inv, scale=1 / 255.0),
            cv2.multiply(layer, a, scale=1 / 255.0),
        )
        out[y0:y1, x0:x1] = blended
        return out

    def to_bgra(self) -> np.ndarray:
        """Drawing with transparent background."""
        bgra = np.dstack([self._layer, self._mask])
        return bgra

    def content_bbox(self, padding: int = 24) -> tuple[int, int, int, int] | None:
        """Bounding box of drawing."""
        if self._bounds is None:
            return None
        bx0, by0, bx1, by1 = self._bounds
        x0 = max(0, bx0 - padding)
        y0 = max(0, by0 - padding)
        x1 = min(self.width, bx1 + padding + 1)
        y1 = min(self.height, by1 + padding + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def cropped_bgra(self, padding: int = 24) -> np.ndarray | None:
        """Cropped transparent drawing."""
        box = self.content_bbox(padding)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return self.to_bgra()[y0:y1, x0:x1]

    def save(self, tag: str = "") -> list[Path]:
        """Save canvas as PNGs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        saved: list[Path] = []

        transparent = self.output_dir / f"drawing_{stamp}{suffix}.png"
        cv2.imwrite(str(transparent), self.to_bgra())
        saved.append(transparent)

        white = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        alpha = (self._mask.astype(np.float32) / 255.0)[:, :, None]
        flat = (white * (1.0 - alpha) + self._layer * alpha).astype(np.uint8)
        on_white = self.output_dir / f"drawing_{stamp}{suffix}_white.png"
        cv2.imwrite(str(on_white), flat)
        saved.append(on_white)

        return saved
