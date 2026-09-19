#!/bin/sh
set -e

# Model weights are already baked into the image (see Dockerfile), so this
# is just starting the local Ollama server, not downloading anything.
ollama serve &

until ollama list >/dev/null 2>&1; do
  sleep 1
done

exec uvicorn src.serve:app --host 0.0.0.0 --port 8000
