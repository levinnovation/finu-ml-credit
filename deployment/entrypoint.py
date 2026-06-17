#!/usr/bin/env python3
"""Railway entrypoint for finu-ml-credit."""

import os
import subprocess
import sys

PORT = os.environ.get("PORT", "8000")
print(f"[finu-ml-credit] Starting FastAPI on :{PORT}", flush=True)
cmd = [
    sys.executable, "-m", "uvicorn", "main:app",
    "--host", "0.0.0.0", "--port", PORT,
    "--log-level", "info",
]
sys.exit(subprocess.call(cmd))
