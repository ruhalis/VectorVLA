"""VLM navigation policy for DreamPilot (see PIVOT.md section 3).

frame (downscaled JPEG) + command + last actions -> strict JSON action state,
validated against the enums in reactor_client. On ANY failure (API error,
timeout, bad JSON, bad enum) the policy holds its previous state — persistent
actions make "do nothing" safe — and retries next tick.

Hard rules implemented here:
  - ONE frame path: frame_to_data_url() is the only downscale+JPEG+base64 code;
    both the offline gate (file paths) and the live loop (numpy arrays) use it.
  - Enforced JSON via forced tool-calling (not parse-and-pray), then enum
    validation on top.
  - Text memory only: the last few actions + one-line reasonings; vision is the
    latest frame only, no frame history.
  - The fallback is a mode, not a rewrite: mode="fallback" swaps the schema to
    "landmark visible? left/center/right?" plus a tiny deterministic controller.

Provider-agnostic: OpenAI-compatible client; VLM_MODEL / VLM_API_KEY /
VLM_BASE_URL from env (VLM_API_KEY falls back to OPENAI_API_KEY).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from reactor_client import LOOKS_H, LOOKS_V, MOVEMENTS

logger = logging.getLogger("vectorvla.policy")

AXES = ("movement", "look_horizontal", "look_vertical")
ENUMS = {"movement": MOVEMENTS, "look_horizontal": LOOKS_H, "look_vertical": LOOKS_V}
IDLE_STATE = {axis: "idle" for axis in AXES}


# ------------------------------------------------------------------ env / client

def load_env(path: Union[str, Path] = ".env") -> None:
    """Set os.environ from a .env file for variables not already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def make_vlm_client():
    """OpenAI-compatible client; swap providers by changing env vars only."""
    from openai import OpenAI

    api_key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set VLM_API_KEY or OPENAI_API_KEY (e.g. in .env)")
    base_url = os.environ.get("VLM_BASE_URL") or None
    # Short timeout + one retry: a hung call must not stall the control loop;
    # holding the previous action state is always safe.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1)


# ------------------------------------------------------------------ frame path

def frame_to_data_url(frame: Union[np.ndarray, str, Path], width: Optional[int] = None) -> str:
    """THE single downscale+JPEG+base64 path (PIVOT hard rule — no second copy).

    Accepts a live (H, W, 3) uint8 RGB array or a recorded frame's file path.
    """
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


# ------------------------------------------------------------------ schemas

def _action_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "set_action",
            "description": "Set the persistent action state for the next ~2 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movement": {"type": "string", "enum": list(MOVEMENTS)},
                    "look_horizontal": {"type": "string", "enum": list(LOOKS_H)},
                    "look_vertical": {"type": "string", "enum": list(LOOKS_V)},
                    "arrived": {"type": "boolean"},
                    "reasoning": {"type": "string", "description": "one short line"},
                },
                "required": ["movement", "look_horizontal", "look_vertical", "arrived", "reasoning"],
            },
        },
    }


def _landmark_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "report_landmark",
            "description": "Report whether the target is visible and where.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visible": {"type": "boolean"},
                    "side": {"type": "string", "enum": ["left", "center", "right"]},
                    "arrived": {"type": "boolean"},
                    "reasoning": {"type": "string", "description": "one short line"},
                },
                "required": ["visible", "side", "arrived", "reasoning"],
            },
        },
    }


FULL_SYSTEM_PROMPT = """\
You are the navigation policy of an agent embodied in a real-time generated \
photoreal 3D world. Each turn you see the agent's current first-person view \
and must set the action state that persists for the next ~2 seconds.

Physics of this body:
- movement translates the camera: forward/back/strafe_left/strafe_right. \
Strafing slides sideways and does NOT turn.
- look_horizontal left/right is how you TURN (there is no other way to turn). \
look_vertical tilts the view up/down; keep it idle unless the target is far \
above or below center.
- Actions take ~1.5 s to visibly take effect, so the current frame may not yet \
show the result of your last action. Do not reverse your last decision unless \
the view clearly demands it.

Strategy:
- If the target is visible and off-center, turn toward it (look_horizontal) \
while moving forward.
- Once it is roughly centered, stop turning (look_horizontal=idle) and keep \
moving forward.
- NEVER strafe to line up with a distant target — turning is how you aim. \
Strafing is only for sidestepping an obstacle directly in front of you.
- If the target just disappeared while you were closing in on it, it is \
probably barely off-frame: turn toward the side where it was last seen \
(see your recent actions), do not start a full search spin.
- If the target is NOT visible and was not seen recently, movement=idle and \
turn steadily in ONE direction until it appears (check your recent actions \
and keep the same turn direction — do not oscillate).
- Set arrived=true when you are within a few meters of the target: it \
dominates the view — or, for a large target like a building, its wall \
fills most of the frame and almost no ground remains between you and it. \
Then use movement=idle.

Call set_action exactly once with all fields. reasoning = one short line.
"""

FALLBACK_SYSTEM_PROMPT = """\
You are the eyes of a navigation agent inside a 3D world, looking at its \
first-person view. Answer ONLY about the target the command names:
- visible: is it in view at all?
- side: is its center in the left, center, or right third of the image? \
(If not visible, give your best guess from your recent reports.)
- arrived: does the target dominate the view (within a few meters)?

Call report_landmark exactly once with all fields. reasoning = one short line.
"""


# ------------------------------------------------------------------ policy

@dataclass
class Decision:
    movement: str
    look_horizontal: str
    look_vertical: str
    arrived: bool
    reasoning: str
    ok: bool            # False -> VLM failed this tick, state was held
    latency_s: float
    raw: Optional[dict] = None

    @property
    def action(self) -> dict:
        return {axis: getattr(self, axis) for axis in AXES}

    def line(self) -> str:
        flag = "" if self.ok else " [HELD]"
        arrived = " ARRIVED" if self.arrived else ""
        return (f"move={self.movement:<12} look_h={self.look_horizontal:<5} "
                f"look_v={self.look_vertical:<4} {self.latency_s:4.1f}s"
                f"{arrived}{flag} | {self.reasoning}")


class VLMPolicy:
    """One instance per episode (per command). decide() is sync and blocking —
    the live loop calls it via asyncio.to_thread, never on the SDK event loop."""

    def __init__(self, command: str, mode: str = "full",
                 client: Any = None, model: Optional[str] = None, history_len: int = 4):
        assert mode in ("full", "fallback"), mode
        self.command = command
        self.mode = mode
        self.client = client or make_vlm_client()
        self.model = model or os.environ.get("VLM_MODEL", "gpt-4o")
        self.state = dict(IDLE_STATE)
        self.history: deque = deque(maxlen=history_len)
        self.calls = 0
        self.failures = 0

    # ---- prompt assembly

    def _messages(self, data_url: str) -> tuple[str, list[dict]]:
        system = FULL_SYSTEM_PROMPT if self.mode == "full" else FALLBACK_SYSTEM_PROMPT
        if self.history:
            memory = "Recent actions, oldest first:\n" + "\n".join(self.history)
        else:
            memory = "This is your first decision of the episode."
        content = [
            {"type": "text", "text": f"Command: {self.command}\n{memory}\nCurrent view:"},
            {"type": "image_url", "image_url": {
                "url": data_url,
                "detail": os.environ.get("VLM_IMAGE_DETAIL", "high"),
            }},
        ]
        return system, [{"role": "user", "content": content}]

    # ---- VLM call (enforced JSON via forced tool call)

    def _call_vlm(self, data_url: str) -> dict:
        tool = _action_tool() if self.mode == "full" else _landmark_tool()
        name = tool["function"]["name"]
        system, messages = self._messages(data_url)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": name}},
            max_tokens=300,
            temperature=0,
        )
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise ValueError("no tool call in response")
        return json.loads(calls[0].function.arguments)

    # ---- validation / fallback controller

    @staticmethod
    def _validate_action(args: dict) -> dict:
        out = {}
        for axis in AXES:
            value = args.get(axis)
            if value not in ENUMS[axis]:
                raise ValueError(f"invalid {axis}={value!r}")
            out[axis] = value
        out["arrived"] = bool(args.get("arrived", False))
        out["reasoning"] = str(args.get("reasoning", ""))[:200]
        return out

    @staticmethod
    def _controller(args: dict) -> dict:
        """Fallback mode: deterministic actions from the VQA answer."""
        visible = bool(args.get("visible", False))
        side = args.get("side")
        if side not in ("left", "center", "right"):
            raise ValueError(f"invalid side={side!r}")
        if not visible:
            action = {"movement": "idle", "look_horizontal": "right", "look_vertical": "idle"}
        else:
            look = {"left": "left", "center": "idle", "right": "right"}[side]
            action = {"movement": "forward", "look_horizontal": look, "look_vertical": "idle"}
        action["arrived"] = bool(args.get("arrived", False))
        action["reasoning"] = str(args.get("reasoning", ""))[:200]
        return action

    # ---- main entry

    def decide(self, frame: Union[np.ndarray, str, Path]) -> Decision:
        t0 = time.monotonic()
        self.calls += 1
        try:
            args = self._call_vlm(frame_to_data_url(frame))
            parsed = self._validate_action(args) if self.mode == "full" else self._controller(args)
        except Exception as e:  # noqa: BLE001 — any failure means: hold state
            self.failures += 1
            logger.warning("VLM decide failed (%s: %s); holding previous state", type(e).__name__, e)
            return Decision(**self.state, arrived=False,
                            reasoning=f"held previous state ({type(e).__name__})",
                            ok=False, latency_s=time.monotonic() - t0)
        arrived = parsed.pop("arrived")
        reasoning = parsed.pop("reasoning")
        self.state = parsed
        self.history.append(
            f"- move={parsed['movement']} look_h={parsed['look_horizontal']}"
            f" look_v={parsed['look_vertical']} arrived={arrived} | {reasoning}"
        )
        return Decision(**parsed, arrived=arrived, reasoning=reasoning,
                        ok=True, latency_s=time.monotonic() - t0, raw=args)
