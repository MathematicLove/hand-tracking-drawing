"""Startup mode menu."""

from __future__ import annotations

import cv2
import numpy as np

import ui

WINDOW = "Hand Tracking Drawing"
ITEMS = (
    ("2d", "2D Drawing"),
    ("3d", "3D Drawing"),
    ("poses", "Poses"),
    ("translater", "Translater"),
)
W, H = 720, 480
TOP = 150
ROW_H = 56
PAD = 60


class Menu:
    """Mode selection screen."""

    def __init__(self) -> None:
        self.index = 0
        self.click = -1
        self.note = ""

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Track hover and clicks."""
        row = (y - TOP) // ROW_H
        if not 0 <= row < len(ITEMS) or not PAD <= x <= W - PAD:
            return
        self.index = int(row)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = int(row)

    def frame(self) -> np.ndarray:
        """Render menu image."""
        img = np.zeros((H, W, 3), dtype=np.uint8)
        ui.put_text(img, "HAND TRACKING DRAWING", (PAD, 54), 26, (255, 255, 255))
        ui.put_text(img, "Select a mode", (PAD, 96), 17, (150, 150, 150))
        for i, (_, label) in enumerate(ITEMS):
            y = TOP + i * ROW_H
            box = (PAD, y, W - PAD, y + ROW_H - 12)
            if i == self.index:
                cv2.rectangle(img, box[:2], box[2:], (45, 45, 45), -1)
                cv2.rectangle(img, box[:2], box[2:], (200, 200, 200), 1, cv2.LINE_AA)
                color = (255, 255, 255)
            else:
                cv2.rectangle(img, box[:2], box[2:], (70, 70, 70), 1, cv2.LINE_AA)
                color = (185, 185, 185)
            ui.put_text(img, f"{i + 1}) {label}", (PAD + 20, y + 12), 19, color)
        ui.put_text(img, "1..4 select    W S move    ENTER start    Q quit",
                    (PAD, H - 70), 16, (140, 140, 140))
        if self.note:
            ui.put_text(img, self.note, (PAD, H - 42), 16, (120, 120, 230))
        return img

    def run(self) -> str | None:
        """Show menu and return mode."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.frame())
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None

            chosen = -1
            if self.click >= 0:
                chosen, self.click = self.click, -1
            elif key in (ord("q"), 27):
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
                return None
            elif key in (ord("w"), 82):
                self.index = (self.index - 1) % len(ITEMS)
            elif key in (ord("s"), 84):
                self.index = (self.index + 1) % len(ITEMS)
            elif key in (13, 10, 32):
                chosen = self.index
            elif ord("1") <= key <= ord("0") + len(ITEMS):
                chosen = key - ord("1")

            if chosen < 0:
                continue
            self.index = chosen
            name = ITEMS[chosen][0]
            if name == "2d":
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
                return name
            self.note = f"{ITEMS[chosen][1]} is not available yet"
