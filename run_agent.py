"""Compatibility shim — the runner lives in the dreampilot package now.

Kept so the demo command `.venv/bin/python run_agent.py --world ...` keeps
working; identical to `python -m dreampilot`.
"""

import asyncio

from dreampilot.runner import main

if __name__ == "__main__":
    asyncio.run(main())
