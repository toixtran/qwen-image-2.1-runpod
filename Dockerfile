# PyTorch 2.7.1 + CUDA 12.8 (Qwen-Image-2.1 needs torch>=2.4)
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: bake the ~33 GB of weights into the image to avoid downloading on cold start.
#   docker build --build-arg BAKE_MODEL=true -t qwen-image-2.1-runpod .
ARG BAKE_MODEL=false
ARG HF_MODEL=Qwen/Qwen-Image-2.1
RUN if [ "$BAKE_MODEL" = "true" ]; then \
      HF_HOME=/models/huggingface python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_MODEL}')"; \
    fi
# handler.py uses /models/huggingface automatically when it exists.

COPY handler.py test_input.json ./

CMD ["python", "-u", "handler.py"]
