"""
video_loader.py
================
Handles opening, reading, and querying metadata from a parking-lot video.

Improvements vs v1
------------------
* **Loop without re-open** — added ``loop_frames`` generator that rewinds
  the capture to frame 0 at end-of-stream instead of closing and re-opening
  the file.  This avoids the OS-level file-open overhead on every loop
  iteration and keeps the capture object alive.
* **Frame-skip iterator** — ``iter_with_skip(n)`` yields every *n*-th frame
  so the caller never has to count frames manually.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class VideoMeta:
    """Metadata extracted from the video capture."""
    width:       int
    height:      int
    fps:         float
    frame_count: int
    path:        str


class VideoLoader:
    """Wrapper around ``cv2.VideoCapture`` with metadata, iteration, and looping.

    Example — basic iteration::

        with VideoLoader("data/parking_video.mp4") as loader:
            for frame in loader:
                process(frame)

    Example — looping without re-opening the file::

        with VideoLoader("data/parking_video.mp4") as loader:
            for frame in loader.loop_frames():
                if done:
                    break
    """

    def __init__(self, video_path: str) -> None:
        """Open the video file and read its metadata.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError:      If OpenCV cannot open the file.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        self.meta = VideoMeta(
            width      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height     = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps        = self._cap.get(cv2.CAP_PROP_FPS) or 25.0,
            frame_count= int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            path       = str(path.resolve()),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def read_frame(self) -> Optional[np.ndarray]:
        """Read the next frame.

        Returns:
            BGR numpy array, or ``None`` at end-of-stream.
        """
        ret, frame = self._cap.read()
        return frame if ret else None

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index (0-based).

        Args:
            frame_index: Target frame position.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))

    def grab_frame(self, frame_index: int = 0) -> Optional[np.ndarray]:
        """Seek to *frame_index* and return that single frame.

        Args:
            frame_index: Frame to grab (default: 0).

        Returns:
            BGR numpy array or ``None`` on failure.
        """
        self.seek(frame_index)
        return self.read_frame()

    def release(self) -> None:
        """Release the underlying ``VideoCapture``."""
        if self._cap.isOpened():
            self._cap.release()

    # ── iterators ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield frames until end-of-stream (single pass)."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def loop_frames(self) -> Iterator[np.ndarray]:
        """Yield frames indefinitely, rewinding at end-of-stream.

        Unlike the pattern ``while True: with VideoLoader(...) as ldr: …``
        this rewinds the existing capture with ``CAP_PROP_POS_FRAMES = 0``
        instead of closing and reopening the file — eliminating the file-open
        overhead on every loop iteration.

        Yields:
            BGR numpy arrays, frame by frame, looping forever.
        """
        while True:
            frame = self.read_frame()
            if frame is None:
                self.seek(0)          # rewind — no file re-open
                frame = self.read_frame()
                if frame is None:
                    break             # truly empty / broken file
            yield frame

    def iter_with_skip(self, skip: int = 1) -> Iterator[np.ndarray]:
        """Yield every *skip*-th frame from the current position.

        Useful for capping detection rate without changing the main loop.

        Args:
            skip: Step size.  ``1`` = every frame; ``3`` = every third frame.

        Yields:
            BGR numpy arrays.
        """
        idx = 0
        for frame in self:
            if idx % skip == 0:
                yield frame
            idx += 1

    # ── context-manager protocol ───────────────────────────────────────────────

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        m = self.meta
        return (
            f"VideoLoader({m.path!r}, "
            f"{m.width}×{m.height}, "
            f"{m.fps:.1f} fps, "
            f"{m.frame_count} frames)"
        )
