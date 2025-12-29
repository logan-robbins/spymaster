"""
Ingestor Service Entry Point (Databento DBN replay).

Use this entry point for historical replay into NATS for downstream
Bronze/Silver processing.
"""

import asyncio

from src.ingestor.replay_publisher import main as replay_main


def main() -> None:
    """Entry point for replay publisher."""
    asyncio.run(replay_main())


if __name__ == "__main__":
    main()
