"""LingBot action vocabulary (SDK_NOTES.md section 3) — shared by body and brain.

Three independent axes, each persistent server-side state applied at chunk
boundaries. Movement is strafe-only; look_horizontal is the ONLY way to turn.
Both reactor_client (sending) and the policies (deciding/validating) import
these — the policy package stays free of any reactor_sdk dependency.
"""

MOVEMENTS = ("idle", "forward", "back", "strafe_left", "strafe_right")
LOOKS_H = ("idle", "left", "right")
LOOKS_V = ("idle", "up", "down")

AXES = ("movement", "look_horizontal", "look_vertical")
ENUMS = {"movement": MOVEMENTS, "look_horizontal": LOOKS_H, "look_vertical": LOOKS_V}
IDLE_STATE = {axis: "idle" for axis in AXES}
