"""Full navigation mode: the VLM decides the action state directly."""

from __future__ import annotations

from dreampilot.actions import AXES, ENUMS, LOOKS_H, LOOKS_V, MOVEMENTS
from dreampilot.policy.base import Policy

SYSTEM_PROMPT = """\
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


class NavigatorPolicy(Policy):
    system_prompt = SYSTEM_PROMPT

    def tool(self) -> dict:
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
                    "required": ["movement", "look_horizontal", "look_vertical",
                                 "arrived", "reasoning"],
                },
            },
        }

    def interpret(self, args: dict) -> dict:
        out = {}
        for axis in AXES:
            value = args.get(axis)
            if value not in ENUMS[axis]:
                raise ValueError(f"invalid {axis}={value!r}")
            out[axis] = value
        out["arrived"] = bool(args.get("arrived", False))
        out["reasoning"] = str(args.get("reasoning", ""))[:200]
        return out
