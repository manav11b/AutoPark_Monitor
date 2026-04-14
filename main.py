"""
main.py
========
Smart Parking Detection System — Aerial/Drone/Satellite Edition.

Redesigned for top-down camera views where YOLO fails (no perspective cues).
Uses a pure pixel-based 4-signal aerial detector — no GPU, no YOLO, no
internet download needed beyond basic Python packages.

Usage
-----
    python main.py [OPTIONS]

Controls
--------
    Q / ESC   → quit
    SPACE     → pause / resume
    S         → save frame snapshot
    +/-       → raise/lower occupancy threshold
    R         → reset background model (after lighting change)
    D         → toggle score debug overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.video_loader    import VideoLoader
from src.slot_manager    import SlotManager
from src.visualization   import Visualizer, VisConfig
from src.pixel_detector  import PixelDetectionConfig
from src.aerial_detector import AerialDetectorConfig


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Parking Detection — Aerial/Drone/Satellite View",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",   default="data/parking_video.mp4")
    p.add_argument("--slots",   default="data/slots.json")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Aerial occupancy score threshold (0–1). "
                        "Lower = more sensitive. Start at 0.30 and tune with +/-.")
    p.add_argument("--fg-thresh", type=int, default=18,
                   help="Pixel deviation from background to count as foreground (px)")
    p.add_argument("--skip",    type=int, default=1,
                   help="Run full aerial detector every N frames (pixel pre-filter always runs)")
    p.add_argument("--no-smooth", action="store_true",
                   help="Disable temporal smoothing (lower latency, more flicker)")
    p.add_argument("--no-loop", action="store_true",
                   help="Play video once and exit")
    p.add_argument("--scale",   type=float, default=1.0,
                   help="Display window scale factor")
    p.add_argument("--save-output", default="",
                   help="Save annotated video to this path (e.g. out.mp4)")
    p.add_argument("--debug-scores", action="store_true",
                   help="Show per-slot raw scores on the frame (useful for threshold tuning)")
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║     Smart Parking Detection — Aerial Edition         ║
║     Pure pixel detection  |  No YOLO  |  No GPU     ║
╚══════════════════════════════════════════════════════╝
Controls:
  Q / ESC → Quit      SPACE → Pause      S → Snapshot
  +/-     → Adjust threshold             R → Reset background
  D       → Toggle score debug overlay
""")


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_debug_scores(
    frame: np.ndarray,
    scores: dict[str, float],
    slot_manager: SlotManager,
    threshold: float,
    scale: float,
) -> np.ndarray:
    """Overlay raw score values on each slot for threshold calibration."""
    overlay = frame.copy()
    for slot_id, sc in scores.items():
        cx, cy = slot_manager.slot_centroid(slot_id)
        cx = int(cx * scale)
        cy = int(cy * scale)
        color = (0, 255, 0) if sc < threshold else (0, 0, 255)
        text  = f"{sc:.2f}"
        cv2.putText(overlay, text, (cx - 18, cy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


class RollingFPS:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print_banner()

    for label, path in [("Video", args.video), ("Slots", args.slots)]:
        if not Path(path).exists():
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    # ── Build components ──────────────────────────────────────────────────────
    pixel_cfg  = PixelDetectionConfig(density_ratio=0.12)   # low gate for aerial
    aerial_cfg = AerialDetectorConfig(
        threshold=args.threshold,
        fg_thresh_px=args.fg_thresh,
    )

    slot_manager = SlotManager(
        slots_path=args.slots,
        pixel_config=pixel_cfg,
        aerial_config=aerial_cfg,
        smoothing=not args.no_smooth,
    )

    visualizer = Visualizer(slot_manager.slots, VisConfig())

    print(f"[INFO] Slots: {slot_manager.total_slots}  |  "
          f"Threshold: {args.threshold}  |  "
          f"FG px delta: {args.fg_thresh}")
    print(f"[INFO] Skip: every {args.skip} frame(s)  |  "
          f"Smoothing: {'off' if args.no_smooth else 'on'}")
    print(f"[INFO] Tip: press D to see per-slot scores and tune threshold with +/-")
    print()

    loader = VideoLoader(args.video)
    meta   = loader.meta
    fps_   = meta.fps or 25.0

    writer: cv2.VideoWriter | None = None
    if args.save_output:
        h = int(meta.height * args.scale)
        w = int(meta.width  * args.scale)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps_, (w, h))
        print(f"[INFO] Recording → '{args.save_output}'")

    paused       = False
    show_debug   = args.debug_scores
    frame_count  = 0
    fps_meter    = RollingFPS()
    t_start      = time.perf_counter()
    delta_step   = 0.02
    last_scores: dict[str, float] = {}

    frame_source = loader.loop_frames() if not args.no_loop else iter(loader)

    try:
        for frame in frame_source:

            while paused:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 == ord(" "):
                    paused = False
                elif k2 in (ord("q"), 27):
                    raise StopIteration

            # ── Detection ─────────────────────────────────────────────────────
            slot_manager.process_frame(frame, skip_interval=args.skip)

            # Compute raw scores every 5 frames for debug overlay
            if show_debug and frame_count % 5 == 0:
                last_scores = slot_manager.slot_scores(frame)

            # ── Visualise ─────────────────────────────────────────────────────
            annotated = visualizer.render(
                frame,
                slot_manager.slots_status,
                slot_manager.available_slots,
                slot_manager.occupied_count,
            )
            display = resize_frame(annotated, args.scale)

            if show_debug and last_scores:
                display = draw_debug_scores(
                    display, last_scores, slot_manager,
                    slot_manager.current_threshold, args.scale,
                )
                # Show current threshold in top-right
                cv2.putText(display,
                    f"THR:{slot_manager.current_threshold:.2f}",
                    (display.shape[1] - 110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,100), 1, cv2.LINE_AA)

            # FPS overlay
            live_fps = fps_meter.tick()
            cv2.putText(display, f"{live_fps:.1f} fps",
                        (display.shape[1] - 85, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow("Smart Parking — Aerial Edition", display)
            if writer:
                writer.write(display)

            frame_count += 1

            if frame_count % 30 == 0:
                avail = ", ".join(slot_manager.available_slots) or "None"
                print(f"  Frame {frame_count:5d} | "
                      f"Occ {slot_manager.occupied_count}/{slot_manager.total_slots} | "
                      f"Avail [{avail}] | thr={slot_manager.current_threshold:.2f} | "
                      f"{live_fps:.1f} fps")

            # ── Keys ──────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = True
                print("[INFO] Paused.")
            elif key == ord("s"):
                snap = f"snapshot_{frame_count:05d}.png"
                cv2.imwrite(snap, display)
                print(f"[INFO] Saved: {snap}")
            elif key == ord("+"):
                t = slot_manager.adjust_threshold(+delta_step)
                print(f"[INFO] Threshold → {t:.2f}  (fewer detections)")
            elif key == ord("-"):
                t = slot_manager.adjust_threshold(-delta_step)
                print(f"[INFO] Threshold → {t:.2f}  (more sensitive)")
            elif key == ord("r"):
                slot_manager.reset_background()
                print("[INFO] Background model reset (adapts over next few frames).")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[INFO] Score debug overlay: {'ON' if show_debug else 'OFF'}")

    except StopIteration:
        pass
    finally:
        loader.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()

    _print_summary(slot_manager, frame_count, t_start)


def _print_summary(mgr: SlotManager, frames: int, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    stats   = mgr.detection_stats()
    print(f"""
╔════════════════════════════════════════╗
║           Session Summary              ║
╠════════════════════════════════════════╣
  Frames processed   : {frames}
  Elapsed            : {elapsed:.1f}s
  Average FPS        : {frames/elapsed if elapsed else 0:.1f}
  Pixel calls        : {stats['pixel_calls']}
  Aerial calls       : {stats['aerial_calls']}
  Pre-filter savings : {stats['aerial_savings_pct']:.1f}%
╚════════════════════════════════════════╝
""")


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
