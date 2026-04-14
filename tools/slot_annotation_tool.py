"""
Parking Slot Annotation Tool
=============================
Interactive OpenCV tool to define parking slot polygons from a video frame.

Usage:
    python tools/slot_annotation_tool.py --video data/parking_video.mp4 --output data/slots.json

Controls:
    Left Click  → Add polygon point
    ENTER       → Finish current polygon (prompts for slot ID)
    R           → Reset / redo current polygon
    Q           → Quit and save all slots
    Z           → Undo last point
"""

import cv2
import json
import argparse
import sys
from pathlib import Path
from typing import Optional


# ── State ──────────────────────────────────────────────────────────────────
current_points: list[list[int]] = []
all_slots: dict[str, list[list[int]]] = {}
frame_display: Optional[object] = None   # will hold a numpy array


# ── Drawing helpers ─────────────────────────────────────────────────────────

def draw_state(frame_base):
    """Redraw the base frame with all committed slots + the in-progress polygon."""
    import numpy as np
    canvas = frame_base.copy()

    # --- committed slots ---
    for slot_id, pts in all_slots.items():
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cv2.putText(canvas, slot_id, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- current in-progress polygon ---
    for i, pt in enumerate(current_points):
        cv2.circle(canvas, tuple(pt), 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(canvas, tuple(current_points[i - 1]), tuple(pt), (0, 0, 255), 2)

    # --- instructions overlay ---
    instructions = [
        "Left Click: Add point",
        "ENTER: Finish polygon",
        "R: Reset polygon",
        "Z: Undo last point",
        "Q: Save & quit",
    ]
    for idx, text in enumerate(instructions):
        cv2.putText(canvas, text, (10, 20 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return canvas


def mouse_callback(event, x: int, y: int, flags, param):
    """Record clicked points and refresh display."""
    global frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        frame_display = draw_state(param)
        cv2.imshow("Slot Annotation Tool", frame_display)


# ── Core annotation loop ─────────────────────────────────────────────────────

def run_annotation(video_path: str, output_path: str, frame_index: int = 0) -> None:
    """Main annotation loop.

    Args:
        video_path:   Path to the parking-lot video.
        output_path:  Destination JSON file for slot polygons.
        frame_index:  Which frame to use as the background (default: 0).
    """
    global current_points, frame_display

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    # Seek to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, base_frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read a frame from the video.")
        sys.exit(1)

    # Load any pre-existing slots so the annotator is additive
    output_file = Path(output_path)
    if output_file.exists():
        try:
            with open(output_file) as fh:
                loaded: dict = json.load(fh)
            all_slots.update({k: v for k, v in loaded.items()})
            print(f"[INFO] Loaded {len(all_slots)} existing slot(s) from {output_path}")
        except json.JSONDecodeError:
            print("[WARN] Existing slots file is invalid JSON – starting fresh.")

    frame_display = draw_state(base_frame)

    cv2.namedWindow("Slot Annotation Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Slot Annotation Tool", mouse_callback, base_frame)
    cv2.imshow("Slot Annotation Tool", frame_display)

    print("\n=== Slot Annotation Tool ===")
    print(f"Video : {video_path}")
    print(f"Output: {output_path}")
    print("Controls: Left-click → add point | ENTER → finish polygon")
    print("          R → reset | Z → undo last | Q → save & quit\n")

    while True:
        key = cv2.waitKey(20) & 0xFF

        # ── ENTER: finish current polygon ──
        if key == 13:
            if len(current_points) < 3:
                print("[WARN] A polygon needs at least 3 points. Keep clicking.")
                continue

            # Ask for slot ID in the terminal
            slot_id = input("Enter slot ID (e.g. A1, B3): ").strip().upper()
            if not slot_id:
                print("[WARN] Empty slot ID – polygon discarded.")
                current_points.clear()
            else:
                if slot_id in all_slots:
                    overwrite = input(f"  Slot '{slot_id}' already exists. Overwrite? [y/N]: ")
                    if overwrite.lower() != "y":
                        print("  Skipped.")
                        current_points.clear()
                        frame_display = draw_state(base_frame)
                        cv2.imshow("Slot Annotation Tool", frame_display)
                        continue

                all_slots[slot_id] = [list(p) for p in current_points]
                print(f"  ✓ Slot '{slot_id}' saved with {len(current_points)} points.")
                current_points.clear()

            frame_display = draw_state(base_frame)
            cv2.imshow("Slot Annotation Tool", frame_display)

        # ── R: reset current polygon ──
        elif key == ord("r"):
            current_points.clear()
            frame_display = draw_state(base_frame)
            cv2.imshow("Slot Annotation Tool", frame_display)
            print("[INFO] Current polygon reset.")

        # ── Z: undo last point ──
        elif key == ord("z"):
            if current_points:
                removed = current_points.pop()
                frame_display = draw_state(base_frame)
                cv2.imshow("Slot Annotation Tool", frame_display)
                print(f"[INFO] Removed point {removed}.")

        # ── Q: save and quit ──
        elif key == ord("q"):
            break

        # ── Window closed ──
        if cv2.getWindowProperty("Slot Annotation Tool", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    _save_slots(output_file)


def _save_slots(output_file: Path) -> None:
    """Persist slot definitions to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fh:
        json.dump(all_slots, fh, indent=2)
    print(f"\n[INFO] Saved {len(all_slots)} slot(s) → {output_file}")
    for sid, pts in all_slots.items():
        print(f"  {sid}: {len(pts)} points")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive parking slot polygon annotation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video",  required=True, help="Path to the parking lot video")
    p.add_argument("--output", default="data/slots.json", help="Output JSON path")
    p.add_argument("--frame",  type=int, default=0,
                   help="Frame index to use as annotation background (default: 0)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_annotation(
        video_path=args.video,
        output_path=args.output,
        frame_index=args.frame,
    )
