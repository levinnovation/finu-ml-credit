#!/bin/sh
set -e
echo "Starting FINU ML Credit service..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Files: $(ls)"
echo "Testing imports..."
python -c "import fastapi; print('fastapi OK')"
python -c "from config import settings; print(f'port={settings.port}')"
echo "Starting uvicorn..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
