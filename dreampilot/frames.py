"""THE single downscale+JPEG+base64 frame path (PIVOT hard rule — no second copy).

Both the offline gate (recorded frame file paths) and the live loop (numpy
arrays straight from the ring buffer) build their VLM image input here, so the
policy is gated offline on byte-identical inputs to what it sees live.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


def _encode_jpeg(frame: Union[np.ndarray, str, Path], width: Optional[int], quality: int) -> bytes:
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = Image.open(frame).convert("RGB")
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def frame_to_data_url(frame: Union[np.ndarray, str, Path], width: Optional[int] = None) -> str:
    """Accepts a live (H, W, 3) uint8 RGB array or a recorded frame's file path."""
    width = width or int(os.environ.get("VLM_IMAGE_WIDTH", "800"))
    return "data:image/jpeg;base64," + base64.b64encode(_encode_jpeg(frame, width, 80)).decode()


def frame_to_jpeg(frame: Union[np.ndarray, str, Path], width: Optional[int] = 1152,
                  quality: int = 78) -> bytes:
    """Display encode for the web UI's MJPEG stream — not the VLM prompt path
    (that is frame_to_data_url above); it lives here so every JPEG encode in
    the repo stays in this one module."""
    return _encode_jpeg(frame, width, quality)
