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


def frame_to_data_url(frame: Union[np.ndarray, str, Path], width: Optional[int] = None) -> str:
    """Accepts a live (H, W, 3) uint8 RGB array or a recorded frame's file path."""
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = Image.open(frame).convert("RGB")
    width = width or int(os.environ.get("VLM_IMAGE_WIDTH", "800"))
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
