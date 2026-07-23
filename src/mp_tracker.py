from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np

from hand import Hand


class MediaPipeTracker:
    """MediaPipe hand landmarker."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.4,
        max_hands: int = 2,
        video_mode: bool = True,
        **_ignored,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._mp = mp
        self.device = "cpu"
        self.max_hands = max_hands
        self.video_mode = video_mode

        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ts = -1

    def __call__(self, frame: np.ndarray) -> list[Hand]:
        """Detect hands in frame."""
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self.video_mode:
            ts = max(int((time.perf_counter() - self._t0) * 1000), self._last_ts + 1)
            self._last_ts = ts
            result = self.detector.detect_for_video(image, ts)
        else:
            result = self.detector.detect(image)

        hands: list[Hand] = []
        for i, lms in enumerate(result.hand_landmarks):
            kp = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
            conf = 1.0
            if result.handedness and i < len(result.handedness):
                conf = float(result.handedness[i][0].score)
            box = np.array([kp[:, 0].min(), kp[:, 1].min(),
                            kp[:, 0].max(), kp[:, 1].max()], dtype=np.float64)
            hands.append(Hand(keypoints=kp, scores=np.ones(21), box=box, conf=conf))

        hands.sort(key=lambda x: x.conf, reverse=True)
        return hands[: self.max_hands]

    def close(self) -> None:
        """Release detector."""
        self.detector.close()
