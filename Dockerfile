# Minimal CPU image for the mini-vLLM OpenAI-compatible server.
# Build:  docker build -t minivllm .
# Run:    docker run -p 8000:8000 -e MINIVLLM_PAGED=1 minivllm
# Then:   http://localhost:8000/  (dashboard) · /v1/chat/completions (OpenAI API)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    MINIVLLM_MODEL=Qwen/Qwen3-0.6B \
    MINIVLLM_SLOTS=4

WORKDIR /app

# Install CPU torch first (its own index), then the package.
COPY pyproject.toml README.md LICENSE ./
COPY minivllm ./minivllm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .

EXPOSE 8000
# A volume at /cache/hf persists the downloaded model across runs.
VOLUME ["/cache/hf"]
CMD ["uvicorn", "minivllm.server:app", "--host", "0.0.0.0", "--port", "8000"]
