FROM python:3.12-slim

# The container is self-contained: it bundles Ollama itself and bakes in
# the two models at build time, so `docker run` (or a Hugging Face Space
# built from this same Dockerfile) needs no external Ollama server and no
# model download at startup.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/processed/chunks.json ./data/processed/chunks.json
COPY faiss_index/ ./faiss_index/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Pull models during the build (not at container start) so the resulting
# image starts up in seconds instead of re-downloading ~2.5GB every run.
RUN ollama serve & \
    OLLAMA_PID=$! && \
    sleep 5 && \
    ollama pull llama3.2:3b && \
    ollama pull nomic-embed-text && \
    kill $OLLAMA_PID

ENV OLLAMA_BASE_URL=http://localhost:11434

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
