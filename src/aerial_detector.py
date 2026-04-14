from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class AerialDetectorConfig:

    threshold: float = 0.38
    fg_thresh_px: int = 22
    fg_weight: float = 0.45
    std_weight: float = 0.30
    sat_weight: float = 0.05
    edge_weight: float = 0.20
    bg_alpha: float = 0.05


class AerialDetector:

    def __init__(self, slot_ids, config=None):

        self._cfg = config or AerialDetectorConfig()

        self._bg = {sid: None for sid in slot_ids}
        self._prev = {sid: None for sid in slot_ids}

    def detect(self, slot_id, crop, crop_mask):

        s = self.score(slot_id, crop, crop_mask)

        decision = "occupied" if s >= self._cfg.threshold else "empty"

        if decision == "empty":
            self._update_bg(slot_id, crop, crop_mask)

        return decision


    def score(self, slot_id, crop, crop_mask):

        if crop is None:
            return 0.0

        if np.count_nonzero(crop_mask) == 0:
            return 0.0


        # LAB brightness channel
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        grey = lab[:, :, 0].astype(np.float32)

        interior = grey[crop_mask == 255]

        if interior.size == 0:
            return 0.0


        bg = self._ensure_bg(slot_id, interior)


        # foreground deviation (shadow-safe)
        diff = np.abs(interior - bg)

        fg_ratio = np.mean(diff > self._cfg.fg_thresh_px)


        # std deviation
        std_score = min(np.std(interior) / 50.0, 1.0)


        # saturation (low weight)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        sat_score = min(
            np.mean(hsv[:, :, 1][crop_mask == 255]) / 80.0,
            1.0
        )


        # edge density
        blurred = cv2.GaussianBlur(grey.astype(np.uint8), (5, 5), 0)

        kernel = np.ones((3, 3), np.uint8)

        blurred = cv2.morphologyEx(
            blurred,
            cv2.MORPH_CLOSE,
            kernel
        )

        edges = cv2.Canny(blurred, 40, 100)

        edge_score = min(
            np.mean(edges[crop_mask == 255] > 0) / 0.20,
            1.0
        )


        # motion detection bonus
        prev = self._prev[slot_id]

        motion_score = 0.0

        if prev is not None:

            diff_frame = cv2.absdiff(prev, grey.astype(np.uint8))

            motion_score = np.mean(diff_frame > 12)

        self._prev[slot_id] = grey.astype(np.uint8)


        combined = (

            self._cfg.fg_weight * fg_ratio +
            self._cfg.std_weight * std_score +
            self._cfg.sat_weight * sat_score +
            self._cfg.edge_weight * edge_score +
            0.15 * motion_score
        )

        return float(np.clip(combined, 0.0, 1.0))


    def _ensure_bg(self, slot_id, interior):

        if self._bg[slot_id] is None:

            self._bg[slot_id] = float(np.mean(interior))

        return self._bg[slot_id]


    def _update_bg(self, slot_id, crop, crop_mask):

        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]

        interior = grey[crop_mask == 255]

        if interior.size == 0:
            return

        current = np.mean(interior)

        old = self._bg[slot_id]

        self._bg[slot_id] = (

            current if old is None else

            (1 - self._cfg.bg_alpha) * old +

            self._cfg.bg_alpha * current
        )
    @property
    def threshold(self):

        return self._cfg.threshold


    @threshold.setter
    def threshold(self, value):

        if not 0.0 < value < 1.0:

            raise ValueError("threshold must be between 0 and 1")

        self._cfg.threshold = value