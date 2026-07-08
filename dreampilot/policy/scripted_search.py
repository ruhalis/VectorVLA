"""Fallback mode: the VLM only answers "landmark visible? left/center/right?"
and a tiny deterministic controller turns that into actions."""

from __future__ import annotations

from dreampilot.policy.base import ADJUST_PULSE_S, Policy

SYSTEM_PROMPT = """\
You are the eyes of a navigation agent inside a 3D world, looking at its \
first-person view. Answer ONLY about the target the command names:
- visible: is it in view at all?
- side: is its center in the left, center, or right third of the image? \
(If not visible, give your best guess from your recent reports.)
- arrived: does the target dominate the view (within a few meters)?

Call report_landmark exactly once with all fields. reasoning = one short line.
"""


class ScriptedSearchPolicy(Policy):
    system_prompt = SYSTEM_PROMPT

    def tool(self) -> dict:
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

    def interpret(self, args: dict) -> dict:
        visible = bool(args.get("visible", False))
        side = args.get("side")
        if side not in ("left", "center", "right"):
            raise ValueError(f"invalid side={side!r}")
        # One axis per pulse (same motion model as the full mode). Search spins
        # use the full pulse (~30 deg); corrections toward a visible target use
        # the short adjust pulse (~10 deg) — a full turn overshoots and hunts.
        if not visible:
            action = {"movement": "idle", "look_horizontal": "right", "look_vertical": "idle"}
        elif side == "center":
            action = {"movement": "forward", "look_horizontal": "idle", "look_vertical": "idle"}
        else:
            action = {"movement": "idle", "look_horizontal": side, "look_vertical": "idle",
                      "pulse_s": ADJUST_PULSE_S}
        action["arrived"] = bool(args.get("arrived", False))
        action["reasoning"] = str(args.get("reasoning", ""))[:200]
        return action
