"""
slot_manager.py
================
Manages slot definitions and runs the aerial detection pipeline.

Pipeline (YOLO removed entirely):
    1. PixelDetector   — fast 2-signal pre-filter (std-dev + edges)
                         Quick-exits clearly-empty slots with no further work.
    2. AerialDetector  — full 4-signal detector (fg ratio + std + sat + edges)
                         Runs on any slot that passes the pre-filter.
                         Includes an adaptive per-slot background model.
    3. Temporal smoothing — majority vote over recent N frames to suppress
                         single-frame flicker from shadows/reflections.

No GPU, no YOLO, no internet download required.  Runs fast on any CPU.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np

from src.pixel_detector  import PixelDetector, PixelDetectionConfig
from src.aerial_detector import AerialDetector, AerialDetectorConfig
from src.polygon_utils   import (
    create_polygon_mask,
    extract_slot_region,
    polygon_centroid,
    bbox_of_points,
    Points,
)


SlotStatus = Literal["occupied", "empty", "unknown"]
_SMOOTHING_WINDOW = 5


class SlotManager:
    """Loads slot polygons and runs occupancy detection on video frames.

    Two-stage aerial pipeline — no YOLO, no GPU required.

    Example::

        mgr = SlotManager("data/slots.json")
        mgr.process_frame(frame)
        print(mgr.available_slots)
    """

    def __init__(
        self,
        slots_path: str,
        pixel_config:  PixelDetectionConfig | None = None,
        aerial_config: AerialDetectorConfig | None = None,
        smoothing:        bool = True,
        smoothing_window: int  = _SMOOTHING_WINDOW,
    ) -> None:
        """Load slot definitions and initialise detectors.

        Args:
            slots_path:       Path to ``slots.json``.
            pixel_config:     Fast pre-filter tuning.
            aerial_config:    Aerial detector tuning.
            smoothing:        Enable temporal majority-vote smoothing.
            smoothing_window: Frames in the smoothing window (default 5).
        """
        self.slots: dict[str, Points] = self._load_slots(slots_path)
        self.slots_status: dict[str, SlotStatus] = {sid: "unknown" for sid in self.slots}

        self.occupied_count:  int = 0
        self.available_count: int = 0

        slot_ids = list(self.slots.keys())
        self._pixel_det  = PixelDetector(pixel_config)
        self._aerial_det = AerialDetector(slot_ids, aerial_config)

        self._smoothing = smoothing
        self._win_size  = smoothing_window
        self._history: dict[str, deque[str]] = {
            sid: deque(maxlen=smoothing_window) for sid in self.slots
        }

        # Mask caches — computed once (fixed camera)
        self._full_mask_cache: dict[str, np.ndarray] = {}
        self._crop_mask_cache: dict[str, np.ndarray] = {}

        # Diagnostics
        self._pixel_calls  = 0
        self._aerial_calls = 0
        self._frame_index  = 0

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        skip_interval: int = 1,
    ) -> dict[str, SlotStatus]:
        """Analyse all slots and update :attr:`slots_status`.

        Args:
            frame:         Current BGR video frame.
            skip_interval: Run AerialDetector every N frames.
                           PixelDetector always runs.  Default: 1 (every frame).

        Returns:
            Updated :attr:`slots_status` dict (same reference).
        """
        run_full = (self._frame_index % skip_interval == 0)
        self._frame_index += 1
        occ = avail = 0

        for slot_id, points in self.slots.items():
            full_mask = self._get_full_mask(slot_id, frame.shape, points)
            crop, _   = extract_slot_region(frame, full_mask, points)
            crop_mask = self._get_crop_mask(slot_id, frame.shape, points)

            # Stage 1 — fast pre-filter
            self._pixel_calls += 1
            hint = self._pixel_det.detect(crop, crop_mask)

            if hint == "possible_vehicle" and run_full:
                # Stage 2 — full aerial detector
                self._aerial_calls += 1
                raw: SlotStatus = self._aerial_det.detect(slot_id, crop, crop_mask)
            elif hint == "possible_vehicle":
                # Carry forward last known status on skipped frames
                raw = self.slots_status.get(slot_id, "empty")
                if raw == "unknown":
                    raw = "empty"
            else:
                raw = "empty"

            # Temporal smoothing
            if self._smoothing:
                self._history[slot_id].append(raw)
                votes = sum(1 for s in self._history[slot_id] if s == "occupied")
                final: SlotStatus = (
                    "occupied" if votes > len(self._history[slot_id]) // 2 else "empty"
                )
            else:
                final = raw

            self.slots_status[slot_id] = final
            if final == "occupied":
                occ += 1
            else:
                avail += 1

        self.occupied_count  = occ
        self.available_count = avail
        return self.slots_status

    # ── threshold control (public — no private attribute access needed) ───────

    def adjust_threshold(self, delta: float) -> float:
        """Adjust the AerialDetector threshold by *delta* (clamped to 0.01–0.99).

        Args:
            delta: Positive to raise (fewer detections), negative to lower.

        Returns:
            New threshold value.
        """
        new = float(np.clip(self._aerial_det.threshold + delta, 0.01, 0.99))
        self._aerial_det.threshold = new
        return new

    @property
    def current_threshold(self) -> float:
        """Current AerialDetector threshold."""
        return self._aerial_det.threshold

    def reset_background(self, slot_id: str | None = None) -> None:
        """Reset adaptive background model for one slot or all slots.

        Call this after a major lighting change (lights on/off, clouds).

        Args:
            slot_id: Slot to reset, or ``None`` to reset all.
        """
        self._aerial_det.reset_background(slot_id)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def total_slots(self) -> int:
        return len(self.slots)

    @property
    def available_slots(self) -> list[str]:
        return sorted(sid for sid, s in self.slots_status.items() if s == "empty")

    @property
    def occupied_slots(self) -> list[str]:
        return sorted(sid for sid, s in self.slots_status.items() if s == "occupied")

    def slot_centroid(self, slot_id: str) -> tuple[int, int]:
        return polygon_centroid(self.slots[slot_id])

    def detection_stats(self) -> dict[str, int | float]:
        """Return call counts and efficiency metrics."""
        savings = (
            100.0 * (1 - self._aerial_calls / self._pixel_calls)
            if self._pixel_calls > 0 else 0.0
        )
        return {
            "pixel_calls":      self._pixel_calls,
            "aerial_calls":     self._aerial_calls,
            "aerial_savings_pct": round(savings, 1),
        }

    def slot_scores(self, frame: np.ndarray) -> dict[str, float]:
        """Return raw AerialDetector scores for every slot — useful for threshold tuning.

        Args:
            frame: Current BGR video frame.

        Returns:
            ``{slot_id: score}`` dict with raw float scores.
        """
        scores = {}
        for slot_id, points in self.slots.items():
            full_mask = self._get_full_mask(slot_id, frame.shape, points)
            crop, _   = extract_slot_region(frame, full_mask, points)
            crop_mask = self._get_crop_mask(slot_id, frame.shape, points)
            scores[slot_id] = self._aerial_det.score(slot_id, crop, crop_mask)
        return scores

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_slots(self, path: str) -> dict[str, Points]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Slots file not found: {path}")
        with open(p) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("slots.json must be a JSON object.")
        slots: dict[str, Points] = {}
        for sid, pts in data.items():
            if not isinstance(pts, list) or len(pts) < 3:
                print(f"[SlotManager] WARN — slot '{sid}' has < 3 points, skipped.")
                continue
            slots[sid] = [[int(c) for c in p_] for p_ in pts]
        print(f"[SlotManager] Loaded {len(slots)} slot(s) from '{path}'.")
        return slots

    def _get_full_mask(self, sid: str, shape: tuple, points: Points) -> np.ndarray:
        if sid not in self._full_mask_cache:
            self._full_mask_cache[sid] = create_polygon_mask(shape, points)
        return self._full_mask_cache[sid]

    def _get_crop_mask(self, sid: str, shape: tuple, points: Points) -> np.ndarray:
        if sid not in self._crop_mask_cache:
            full = self._get_full_mask(sid, shape, points)
            x, y, bw, bh = bbox_of_points(points, shape)
            self._crop_mask_cache[sid] = full[y:y + bh, x:x + bw].copy()
        return self._crop_mask_cache[sid]
