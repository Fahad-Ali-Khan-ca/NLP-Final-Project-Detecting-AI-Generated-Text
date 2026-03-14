FROM python:3.10-slim

WORKDIR /app

# System deps for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    torchaudio==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] gradio

# Copy application code
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY app.py .

# Models are mounted at runtime via docker-compose volumes
# COPY models/ models/

EXPOSE 8000 7860

# Default: run the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
