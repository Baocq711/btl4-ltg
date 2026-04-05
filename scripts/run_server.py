import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.server.server import serve


if __name__ == "__main__":
    host = os.getenv("QUORIDOR_HOST", "127.0.0.1")
    port = int(os.getenv("QUORIDOR_PORT", "8765"))
    asyncio.run(serve(host=host, port=port))
