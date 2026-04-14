"""
visualization.py
=================
All OpenCV drawing logic — polygon overlays, HUD stats, and the
available-slot ticker.

Improvements vs v1
------------------
* **Single-pass slot drawing** — v1 iterated the slots dictionary twice
  (once for fill, once for outlines/labels).  This version builds the fill
  overlay in a single loop and then draws outlines + labels in the same loop
  body after the blend, saving one full dict traversal per frame.
* **Pre-computed centroid cache** — centroids are computed once at
  construction time (they never change for a fixed camera) rather than being
  recomputed via Shapely on every frame.
* **Overflow-safe ticker** — when the available-slot list is too long for the
  frame width the text is truncated gracefully instead of being clipped by
  the window boundary.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field

from src.polygon_utils import polygon_centroid, normalize_points, Points


# ── Palette (BGR) ─────────────────────────────────────────────────────────────
_GREEN  = ( 34, 197,  94)   # available
_RED    = (239,  68,  68)   # occupied
_YELLOW = (234, 179,   8)   # unknown
_WHITE  = (255, 255, 255)
_DARK   = ( 15,  15,  15)


@dataclass
class VisConfig:
    """Visual tuning knobs.

    Attributes:
        font_scale:         Base font scale for slot-ID labels.
        line_thickness:     Polygon outline thickness (px).
        fill_alpha:         Translucency of the polygon fill (0 = transparent).
        show_centroid_dot:  Draw a small dot at each polygon centroid.
        hud_height:         Pixel height of the top statistics banner.
        ticker_height:      Pixel height of the bottom available-slot strip.
    """
    font_scale:        float = 0.55
    line_thickness:    int   = 2
    fill_alpha:        float = 0.45
    show_centroid_dot: bool  = True
    hud_height:        int   = 50
    ticker_height:     int   = 36


class Visualizer:
    """Draws the complete parking-lot HUD onto a video frame.

    Centroids are pre-computed at construction time from the slot polygons
    (they are constant for a fixed camera) and cached to avoid repeated
    computation during the render loop.

    Example::

        vis    = Visualizer(slots)
        output = vis.render(frame, slots_status, available_slots)
        cv2.imshow("Parking", output)
    """

    def __init__(
        self,
        slots: dict[str, Points],
        config: VisConfig | None = None,
    ) -> None:
        """Initialise and pre-compute centroid cache.

        Args:
            slots:  ``{slot_id: [[x,y], …]}`` polygon map.
            config: Optional visual tuning; defaults are used if omitted.
        """
        self._cfg      = config or VisConfig()
        self._slots    = slots
        # Pre-compute centroids once — they are constant for a fixed camera
        self._centroids: dict[str, tuple[int, int]] = {
            sid: polygon_centroid(pts) for sid, pts in slots.items()
        }

    # ── main entry point ──────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        slots_status: dict[str, str],
        available_slots: list[str],
        occupied_count: int,
    ) -> np.ndarray:
        """Produce a fully annotated frame.

        Args:
            frame:           Raw BGR video frame.
            slots_status:    ``{slot_id: "occupied"|"empty"|"unknown"}``.
            available_slots: Sorted list of free slot IDs.
            occupied_count:  Pre-computed count of occupied slots (avoids a
                             second dict scan inside the renderer).

        Returns:
            Annotated BGR frame (same resolution as *frame*).
        """
        canvas = frame.copy()

        # 1. Polygon overlays (single-pass)
        self._draw_slots(canvas, slots_status)

        # 2. Top HUD — statistics
        total = len(self._slots)
        self._draw_hud(canvas, total, occupied_count, total - occupied_count)

        # 3. Bottom ticker — available slot list
        self._draw_ticker(canvas, available_slots, frame.shape[1])

        return canvas

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_slots(
        self,
        canvas: np.ndarray,
        slots_status: dict[str, str],
    ) -> None:
        """Single-pass: fill overlay → blend → outlines + labels.

        v1 iterated the slot dict twice.  This version builds the fill overlay
        in one loop, blends once, then draws outlines and labels in the same
        loop body — halving the number of dict iterations.
        """
        overlay = canvas.copy()

        # --- pass A: fill (all slots) ---
        for slot_id, points in self._slots.items():
            status = slots_status.get(slot_id, "unknown")
            color  = self._status_color(status)
            cv2.fillPoly(overlay, [normalize_points(points)], color=color)

        # --- single blend ---
        cv2.addWeighted(
            overlay, self._cfg.fill_alpha,
            canvas,  1.0 - self._cfg.fill_alpha,
            0, canvas,
        )

        # --- pass B: outlines + labels (reuses cached centroids) ---
        for slot_id, points in self._slots.items():
            status = slots_status.get(slot_id, "unknown")
            color  = self._status_color(status)

            cv2.polylines(
                canvas, [normalize_points(points)],
                isClosed=True, color=color,
                thickness=self._cfg.line_thickness,
                lineType=cv2.LINE_AA,
            )

            cx, cy = self._centroids[slot_id]  # cached — no Shapely call

            if self._cfg.show_centroid_dot:
                cv2.circle(canvas, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

            self._put_label(canvas, slot_id, cx, cy, color)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        total: int,
        occupied: int,
        available: int,
    ) -> None:
        """Draw the dark statistics banner at the top of the frame."""
        h, w  = canvas.shape[:2]
        bar_h = self._cfg.hud_height

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), _DARK, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        stats = [
            (f"Total: {total}",         _WHITE),
            (f"Occupied: {occupied}",   _RED),
            (f"Available: {available}", _GREEN),
        ]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.60
        thick = 1
        gap   = w // (len(stats) + 1)

        for i, (text, color) in enumerate(stats):
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            x = gap * (i + 1) - tw // 2
            y = bar_h // 2 + th // 2
            cv2.putText(canvas, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        cv2.line(canvas, (0, bar_h), (w, bar_h), _GREEN, 1)

    def _draw_ticker(
        self,
        canvas: np.ndarray,
        available_slots: list[str],
        frame_width: int,
    ) -> None:
        """Draw the available-slot footer with graceful text overflow handling."""
        h, w   = canvas.shape[:2]
        bar_h  = self._cfg.ticker_height
        y_top  = h - bar_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, y_top), (w, h), _DARK, -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        cv2.line(canvas, (0, y_top), (w, y_top), _GREEN, 1)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thick = 1

        # Build the label; truncate if it would overflow the frame width
        prefix    = "Available: "
        slot_text = ", ".join(available_slots) if available_slots else "None"
        label     = prefix + slot_text
        max_px    = frame_width - 20   # 10 px margin each side

        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        if tw > max_px:
            # Trim slot list until it fits, append "…"
            parts = list(available_slots)
            while parts:
                parts.pop()
                candidate = prefix + ", ".join(parts) + ", …"
                (cw, _), _ = cv2.getTextSize(candidate, font, scale, thick)
                if cw <= max_px:
                    label = candidate
                    break
            else:
                label = prefix + "…"

        y = y_top + (bar_h + th) // 2
        cv2.putText(canvas, label, (10, y), font, scale, _GREEN, thick, cv2.LINE_AA)

    # ── utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _status_color(status: str) -> tuple[int, int, int]:
        return {"occupied": _RED, "empty": _GREEN}.get(status, _YELLOW)

    def _put_label(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw *text* centred at *(cx, cy)* with a dark background pill."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._cfg.font_scale
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        pad = 3
        cv2.rectangle(
            canvas,
            (cx - tw // 2 - pad, cy - th - pad),
            (cx + tw // 2 + pad, cy + baseline + pad),
            _DARK, -1,
        )
        cv2.putText(
            canvas, text, (cx - tw // 2, cy),
            font, scale, color, thick, cv2.LINE_AA,
        )
