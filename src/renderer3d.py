from __future__ import annotations

import cv2
import numpy as np

_FRONT = (0, 1, 2, 3)
_BACK = (5, 4, 7, 6)
_SIDES = (
    (4, 5, 1, 0),
    (3, 2, 6, 7),
    (4, 0, 3, 7),
    (1, 5, 6, 2),
)

_LIGHT = np.array([0.35, -0.55, -1.0])
_LIGHT /= np.linalg.norm(_LIGHT)

_CARD_BG = (40, 37, 34)
_CARD_BORDER = (120, 115, 110)
_EDGE_COLOR = (150, 145, 140)
_SIDE_COLOR = np.array([120, 116, 112], dtype=np.float32)


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build rotation matrix."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Object3D:
    """Textured 3D slab."""

    def __init__(self, drawing_bgra: np.ndarray, thickness_ratio: float = 0.05) -> None:
        self.texture = self._make_card(drawing_bgra)
        h, w = self.texture.shape[:2]
        self.aspect = w / max(h, 1)
        self.thickness_ratio = thickness_ratio
        self._corners = self._build_corners()

    @staticmethod
    def _make_card(drawing_bgra: np.ndarray) -> np.ndarray:
        """Flatten drawing to card."""
        if drawing_bgra.shape[2] == 4:
            rgb = drawing_bgra[:, :, :3].astype(np.float32)
            alpha = (drawing_bgra[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        else:
            rgb = drawing_bgra.astype(np.float32)
            alpha = np.ones(drawing_bgra.shape[:2] + (1,), dtype=np.float32)

        h, w = rgb.shape[:2]
        max_side = 900
        if max(h, w) > max_side:
            k = max_side / max(h, w)
            rgb = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)[:, :, None]

        bg = np.full(rgb.shape, _CARD_BG, dtype=np.float32)
        card = (bg * (1.0 - alpha) + rgb * alpha).astype(np.uint8)
        cv2.rectangle(card, (0, 0), (card.shape[1] - 1, card.shape[0] - 1), _CARD_BORDER, 2)
        return card

    def _build_corners(self) -> np.ndarray:
        """Eight slab vertices."""
        w = self.aspect * 0.5
        h = 0.5
        t = max(self.thickness_ratio, 0.005) * 0.5
        return np.array(
            [
                (-w, -h, t), (w, -h, t), (w, h, t), (-w, h, t),
                (-w, -h, -t), (w, -h, -t), (w, h, -t), (-w, h, -t),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _project(pts3d: np.ndarray, focal: float, distance: float, cx: float, cy: float):
        """Perspective project points."""
        z = pts3d[:, 2] + distance
        z = np.maximum(z, 1e-3)
        x = focal * pts3d[:, 0] / z + cx
        y = focal * pts3d[:, 1] / z + cy
        return np.stack([x, y], axis=1), z

    @staticmethod
    def _shade(face_cam: np.ndarray) -> float:
        """Face lighting factor."""
        n = np.cross(face_cam[1] - face_cam[0], face_cam[2] - face_cam[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return 0.2
        n = n / norm
        if n[2] > 0:
            n = -n
        return float(np.clip(abs(np.dot(n, _LIGHT)), 0.22, 1.0))

    @staticmethod
    def _quad_area(quad: np.ndarray) -> float:
        """Screen area of quad."""
        x, y = quad[:, 0], quad[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _blit_texture(
        self, out: np.ndarray, texture: np.ndarray, quad: np.ndarray, shade: float
    ) -> None:
        """Warp texture onto face."""
        oh, ow = out.shape[:2]
        x0 = int(np.floor(quad[:, 0].min()))
        y0 = int(np.floor(quad[:, 1].min()))
        x1 = int(np.ceil(quad[:, 0].max())) + 1
        y1 = int(np.ceil(quad[:, 1].max())) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(ow, x1), min(oh, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return

        th, tw = texture.shape[:2]
        src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        dst = (quad - np.array([x0, y0], dtype=np.float64)).astype(np.float32)
        try:
            m = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return
        if not np.all(np.isfinite(m)):
            return

        size = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(texture, m, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.full((th, tw), 255, np.uint8), m, size, flags=cv2.INTER_NEAREST
        )
        warped = np.clip(warped.astype(np.float32) * (0.70 + 0.30 * shade), 0, 255).astype(np.uint8)

        roi = out[y0:y1, x0:x1]
        a = (mask.astype(np.float32) / 255.0)[:, :, None]
        out[y0:y1, x0:x1] = (roi * (1.0 - a) + warped * a).astype(np.uint8)

    def render(
        self,
        background: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float = 0.0,
        scale: float = 1.0,
        distance: float = 3.0,
        center: tuple[float, float] | None = None,
        draw_edges: bool = True,
    ) -> np.ndarray:
        """Render object over background."""
        out = background.copy()
        oh, ow = out.shape[:2]
        cx, cy = center if center is not None else (ow * 0.5, oh * 0.5)
        focal = 1.7 * oh * max(scale, 0.05)

        rot = _rotation(yaw, pitch, roll)
        cam = self._corners @ rot.T
        proj, depth = self._project(cam, focal, distance, cx, cy)

        faces: list[tuple[float, tuple[int, ...], str]] = [
            (float(depth[list(_FRONT)].mean()), _FRONT, "front"),
            (float(depth[list(_BACK)].mean()), _BACK, "back"),
        ]
        faces += [(float(depth[list(f)].mean()), f, "side") for f in _SIDES]
        faces.sort(key=lambda item: item[0], reverse=True)

        back_tex = None
        for _, idx, kind in faces:
            centre = cam[list(idx)].mean(axis=0)
            view = centre + np.array([0.0, 0.0, distance])
            if float(np.dot(centre, view)) >= 0.0:
                continue

            quad = proj[list(idx)]
            if self._quad_area(quad) < 4.0:
                continue
            shade = self._shade(cam[list(idx)])

            if kind == "front":
                self._blit_texture(out, self.texture, quad, shade)
            elif kind == "back":
                if back_tex is None:
                    back_tex = cv2.flip(self.texture, 1)
                    back_tex = (back_tex.astype(np.float32) * 0.55).astype(np.uint8)
                self._blit_texture(out, back_tex, quad, shade)
            else:
                color = np.clip(_SIDE_COLOR * shade, 0, 255)
                cv2.fillConvexPoly(
                    out, quad.astype(np.int32), color.tolist(), lineType=cv2.LINE_AA
                )

            if draw_edges:
                cv2.polylines(
                    out, [quad.astype(np.int32)], True, _EDGE_COLOR, 1, cv2.LINE_AA
                )
        return out
