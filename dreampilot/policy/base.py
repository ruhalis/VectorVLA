"""VLM policy base: frame + command + text memory -> validated action state.

Shared machinery for every policy mode:
  - prompt assembly (vision = latest frame only; text memory = the last few
    actions + one-line reasonings — cheap hysteresis against direction hunting),
  - the VLM call with enforced JSON via a forced tool call (not parse-and-pray),
  - hold-previous-state on ANY failure (API error, timeout, bad JSON, bad enum):
    persistent actions make "do nothing" safe; retry next tick,
  - bookkeeping (history, call/failure counts).

A mode is a subclass providing three things: `system_prompt`, `tool()` (the
forced-call schema), and `interpret()` (tool args -> action dict). That keeps
the PIVOT rule "the fallback is a mode, not a rewrite" structural: swapping
policy class swaps schema + controller and nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from dreampilot.actions import AXES, IDLE_STATE
from dreampilot.frames import frame_to_data_url
from dreampilot.vlm import default_model, is_reasoning_model, make_vlm_client

logger = logging.getLogger("vectorvla.policy")

# Turn magnitude = rotation speed x pulse length, so a fine turn is a SHORTER
# pulse on the same axis: the default 2 s pulse swings ~30 deg, which
# overshoots any target that is nearly centered. 0.7 s stays just above one
# chunk (~0.61 s measured in run_001), so the short pulse still spans a chunk
# boundary and reliably takes effect -> roughly 10 deg.
ADJUST_PULSE_S = 0.7


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
    pulse_s: Optional[float] = None  # policy override of the pulse length; None -> runner --period

    @property
    def action(self) -> dict:
        return {axis: getattr(self, axis) for axis in AXES}

    def line(self) -> str:
        flag = "" if self.ok else " [HELD]"
        arrived = " ARRIVED" if self.arrived else ""
        pulse = f" pulse={self.pulse_s:.1f}s" if self.pulse_s is not None else ""
        return (f"move={self.movement:<12} look_h={self.look_horizontal:<5} "
                f"look_v={self.look_vertical:<4} {self.latency_s:4.1f}s"
                f"{pulse}{arrived}{flag} | {self.reasoning}")


class Policy(ABC):
    """One instance per episode (per command). decide() is sync and blocking —
    the live loop calls it via asyncio.to_thread, never on the SDK event loop."""

    system_prompt: str  # set by each subclass

    def __init__(self, command: str, client: Any = None,
                 model: Optional[str] = None, history_len: int = 4):
        self.command = command
        self.client = client or make_vlm_client()
        self.model = model or default_model()
        self.state = dict(IDLE_STATE)
        self.history: deque = deque(maxlen=history_len)
        self.calls = 0
        self.failures = 0

    # ---- the mode: schema + interpretation

    @abstractmethod
    def tool(self) -> dict:
        """OpenAI tool definition the VLM is forced to call."""

    @abstractmethod
    def interpret(self, args: dict) -> dict:
        """Tool args -> {*AXES, "arrived", "reasoning"}; raise on anything invalid.
        May include "action_name" (the mode's own vocabulary) — used only for
        the text-memory history so the model sees its past choices in the same
        terms its prompt uses — and "pulse_s" (a shorter pulse for fine turns;
        None/absent -> the runner's --period)."""

    # ---- prompt assembly

    def _user_text(self) -> str:
        if self.history:
            memory = "Recent actions, oldest first:\n" + "\n".join(self.history)
        else:
            memory = "This is your first decision of the episode."
        return f"Command: {self.command}\n{memory}\nCurrent view:"

    def _messages(self, data_url: str) -> list[dict]:
        content = [
            {"type": "text", "text": self._user_text()},
            {"type": "image_url", "image_url": {
                "url": data_url,
                "detail": os.environ.get("VLM_IMAGE_DETAIL", "high"),
            }},
        ]
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}]

    # ---- VLM call (enforced JSON via forced tool call)

    def _call_vlm(self, data_url: str) -> dict:
        tool = self.tool()
        name = tool["function"]["name"]
        if is_reasoning_model(self.model):
            return self._call_responses(tool, name, data_url)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(data_url),
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": name}},
            max_tokens=300,
            temperature=0,
        )
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise ValueError("no tool call in response")
        return json.loads(calls[0].function.arguments)

    def _call_responses(self, tool: dict, name: str, data_url: str) -> dict:
        """gpt-5.x path: forced function tools + reasoning_effort are only
        served on the Responses API (gpt-5.4 rejects them on chat completions).
        No temperature; reasoning tokens count against max_output_tokens, so
        the cap sits far above the ~100 tokens the tool call itself needs."""
        fn = tool["function"]
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system",
                 "content": [{"type": "input_text", "text": self.system_prompt}]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": self._user_text()},
                    {"type": "input_image", "image_url": data_url,
                     "detail": os.environ.get("VLM_IMAGE_DETAIL", "high")},
                ]},
            ],
            tools=[{"type": "function", "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn["parameters"]}],
            tool_choice={"type": "function", "name": name},
            reasoning={"effort": os.environ.get("VLM_REASONING_EFFORT", "low")},
            max_output_tokens=2000,
        )
        for item in resp.output:
            if getattr(item, "type", None) == "function_call":
                return json.loads(item.arguments)
        raise ValueError("no tool call in response")

    # ---- main entry

    def decide(self, frame: Union[np.ndarray, str, Path]) -> Decision:
        t0 = time.monotonic()
        self.calls += 1
        try:
            args = self._call_vlm(frame_to_data_url(frame))
            parsed = self.interpret(args)
        except Exception as e:  # noqa: BLE001 — any failure means: hold state
            self.failures += 1
            logger.warning("VLM decide failed (%s: %s); holding previous state", type(e).__name__, e)
            return Decision(**self.state, arrived=False,
                            reasoning=f"held previous state ({type(e).__name__})",
                            ok=False, latency_s=time.monotonic() - t0)
        arrived = parsed.pop("arrived")
        reasoning = parsed.pop("reasoning")
        pulse_s = parsed.pop("pulse_s", None)
        label = parsed.pop("action_name", None) or (
            f"move={parsed['movement']} look_h={parsed['look_horizontal']}"
            f" look_v={parsed['look_vertical']}")
        self.state = parsed
        self.history.append(f"- {label} arrived={arrived} | {reasoning}")
        return Decision(**parsed, arrived=arrived, reasoning=reasoning, ok=True,
                        latency_s=time.monotonic() - t0, raw=args, pulse_s=pulse_s)
