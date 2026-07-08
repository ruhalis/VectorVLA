"""Policy modes. A new behavior is a new Policy subclass registered in MODES —
the runner, the offline gate, and the CLI pick it up by name automatically."""

from dreampilot.policy.base import Decision, Policy
from dreampilot.policy.navigator import NavigatorPolicy
from dreampilot.policy.scripted_search import ScriptedSearchPolicy

MODES: dict[str, type[Policy]] = {
    "full": NavigatorPolicy,
    "fallback": ScriptedSearchPolicy,
}


def make_policy(command: str, mode: str = "full", **kwargs) -> Policy:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {sorted(MODES)}")
    return MODES[mode](command, **kwargs)


__all__ = ["Decision", "Policy", "NavigatorPolicy", "ScriptedSearchPolicy",
           "MODES", "make_policy"]
