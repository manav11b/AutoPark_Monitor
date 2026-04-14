from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PixelDetectionConfig:
    blur_kernel: tuple[int, int] = (5, 5)
    density_ratio: float = 0.10
    std_weight: float = 0.65
    edge_weight: float = 0.35
    std_norm_max: float = 50.0


class PixelDetector:

    def __init__(self, config=None):
        self._cfg = config or PixelDetectionConfig()

    def detect(self, slot_crop, crop_mask):

        if slot_crop is None or slot_crop.size == 0:
            return "empty"

        polygon_pixels = np.count_nonzero(crop_mask)

        if polygon_pixels == 0:
            return "empty"

        # LAB brightness instead of grayscale
        lab = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2LAB)
        grey = lab[:, :, 0]

        interior = grey[crop_mask == 255]

        std_score = min(
            np.std(interior) / self._cfg.std_norm_max,
            1.0
        )

        blurred = cv2.GaussianBlur(grey, self._cfg.blur_kernel, 0)

        kernel = np.ones((3, 3), np.uint8)
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(blurred, 40, 100)

        edges_in = edges[crop_mask == 255]

        edge_score = min(
            np.mean(edges_in > 0),
            1.0
        )

        score = (
            self._cfg.std_weight * std_score +
            self._cfg.edge_weight * edge_score
        )

        return "possible_vehicle" if score >= self._cfg.density_ratio else "empty"