"""Provider-agnostic VLM client: swap providers by changing env vars only.

VLM_MODEL / VLM_API_KEY / VLM_BASE_URL from env (VLM_API_KEY falls back to
OPENAI_API_KEY) — switchable in one line if a key dies at the venue.
"""

from __future__ import annotations

import os


def make_vlm_client():
    from openai import OpenAI

    api_key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set VLM_API_KEY or OPENAI_API_KEY (e.g. in .env)")
    base_url = os.environ.get("VLM_BASE_URL") or None
    # Short timeout + one retry: a hung call must not stall the control loop;
    # holding the previous action state is always safe.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1)


def default_model() -> str:
    return os.environ.get("VLM_MODEL", "gpt-4o")
