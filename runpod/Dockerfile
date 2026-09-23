FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*
COPY runpod/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c 'from diffusers import QwenImage21Pipeline; from transformers import Qwen3VLForConditionalGeneration, AutoModelForImageTextToText'

# All application files live outside the RunPod persistent volume mount.
WORKDIR /app
COPY app.py inference_worker.py download_models.py models.lock.json /app/
COPY static/ /app/static/
COPY scripts/ /app/scripts/
COPY runpod/entrypoint.sh /usr/local/bin/playground-entrypoint
RUN chmod +x /usr/local/bin/playground-entrypoint
ENV PYTHONUNBUFFERED=1 \
    PLAYGROUND_BACKEND=diffusers \
    PLAYGROUND_HOST=0.0.0.0 \
    PORT=8765 \
    PLAYGROUND_MODEL_DIR=/workspace/models \
    PLAYGROUND_OUTPUT_DIR=/workspace/outputs \
    HF_HOME=/workspace/huggingface \
    PLAYGROUND_REWRITER=1
LABEL org.opencontainers.image.source="https://github.com/rama-adi/qwen-image-2-1-abliterated-playground"
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60m --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=3)'
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/playground-entrypoint"]
