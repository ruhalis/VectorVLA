"""Full navigation mode: the VLM picks exactly ONE action per decision.

The runner drives this as a pulse: the chosen action runs for ~2 s, then the
runner stops the agent and the next decision is made from a settled frame.
One action at a time (never move + turn together) keeps each pulse's effect
attributable — the model can see what its last action did before choosing.
"""

from __future__ import annotations

from dreampilot.actions import IDLE_STATE
from dreampilot.policy.base import ADJUST_PULSE_S, Policy

# Single-action vocabulary -> the LingBot axis state it maps to (unlisted axes idle).
# adjust_* are the same turn axis at a short pulse (~10 deg vs ~30 deg): fine
# aiming for the final approach, where a full turn overshoots and hunts.
SINGLE_ACTIONS = {
    "turn_left": {"look_horizontal": "left"},
    "turn_right": {"look_horizontal": "right"},
    "adjust_left": {"look_horizontal": "left"},
    "adjust_right": {"look_horizontal": "right"},
    "forward": {"movement": "forward"},
    "back": {"movement": "back"},
    "strafe_left": {"movement": "strafe_left"},
    "strafe_right": {"movement": "strafe_right"},
    "look_up": {"look_vertical": "up"},
    "look_down": {"look_vertical": "down"},
    "stop": {},
}

PULSE_OVERRIDES = {"adjust_left": ADJUST_PULSE_S, "adjust_right": ADJUST_PULSE_S}

SYSTEM_PROMPT = """\
You are the navigation policy of an agent embodied in a real-time generated \
photoreal 3D world. Each turn you see the agent's current first-person view \
and pick EXACTLY ONE action. The action runs for about 2 seconds, then the \
agent stops automatically and you get a fresh, settled view for your next \
decision. You never need to stop an action yourself.

Actions:
- forward / back: walk without turning. forward is how you close distance.
- turn_left / turn_right: a BIG rotation in place (about 30 degrees) — for \
searching, or when the target is out of view or far off to the side.
- adjust_left / adjust_right: a SMALL rotation (about 10 degrees) — for fine \
aiming, mainly during the final approach.
- strafe_left / strafe_right: slide sideways WITHOUT turning — only for \
sidestepping an obstacle directly in front of you. NEVER strafe to line up \
with a distant target; turning is how you aim.
- look_up / look_down: tilt the view — only if the target is far above or \
below center.
- stop: do nothing this turn (use it with arrived=true when you have reached \
the target).

Strategy:
- Target visible anywhere in the frame and still FAR -> forward. Do NOT try \
to center a distant target: a turn swings the view ~30 degrees and will \
overshoot it, and walking closes distance even while it sits off to one side.
- Target far but sliding toward the frame edge -> one adjust toward it, then \
back to forward.
- Target CLOSE (it fills a good part of the view) and off-center -> now aim: \
adjust_left / adjust_right until it is roughly centered, then forward for \
the last stretch. Use turn_left / turn_right here only if it is more than \
about a third of the frame away from center.
- If the target just disappeared while you were closing in, it is probably \
barely off-frame: one adjust toward the side where it was last seen (see \
your recent actions) — do not start a full search spin.
- Target not visible and not seen recently -> keep turning \
(turn_left / turn_right) in ONE consistent direction until it appears \
(check your recent actions; do not oscillate).
- Set arrived=true when you are within a few meters of the target: it \
dominates the view — or, for a large target like a building, its wall fills \
most of the frame and almost no ground remains between you and it. Then \
action=stop.

Each pulse moves or turns you only a small step — expect to repeat forward or \
the same turn several times in a row. Do not reverse your previous action \
unless the view clearly demands it.

Call set_action exactly once. reasoning = one short line.
"""


class NavigatorPolicy(Policy):
    system_prompt = SYSTEM_PROMPT

    def tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "set_action",
                "description": "Pick the single action to run for the next ~2 second pulse.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(SINGLE_ACTIONS)},
                        "arrived": {"type": "boolean"},
                        "reasoning": {"type": "string", "description": "one short line"},
                    },
                    "required": ["action", "arrived", "reasoning"],
                },
            },
        }

    def interpret(self, args: dict) -> dict:
        name = args.get("action")
        if name not in SINGLE_ACTIONS:
            raise ValueError(f"invalid action={name!r}")
        out = dict(IDLE_STATE)
        out.update(SINGLE_ACTIONS[name])
        out["action_name"] = name
        out["pulse_s"] = PULSE_OVERRIDES.get(name)
        out["arrived"] = bool(args.get("arrived", False))
        out["reasoning"] = str(args.get("reasoning", ""))[:200]
        return out
