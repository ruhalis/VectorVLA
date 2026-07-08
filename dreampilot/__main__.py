"""Live DreamPilot runner: python -m dreampilot --world village"""

import asyncio

from dreampilot.runner import main

if __name__ == "__main__":
    asyncio.run(main())
