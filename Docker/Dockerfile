# Base Python image
FROM python:3.9-slim-buster 

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# --- ATTEMPT TO PRELOAD LIBGOMP ---
# The path /usr/lib/aarch64-linux-gnu/libgomp.so.1 comes from the error message inside the container.
# If your system uses a different path for libgomp.so.1 (when inspecting a running container), adjust this.
# For amd64 images, it might be /usr/lib/x86_64-linux-gnu/libgomp.so.1.
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
# --- END OF ATTEMPT TO PRELOAD LIBGOMP ---

# Install system dependencies: git (for cloning), ffmpeg & libsndfile1 (for audio processing)
# build-essential & cmake might be needed by some Python packages during pip install
# Explicitly add libgomp1 for OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \ 
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install specific PyTorch version (CPU for this API) compatible with OpenVoice
RUN pip install --no-cache-dir torch==1.13.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Clone the OpenVoice repository
RUN git clone https://github.com/myshell-ai/OpenVoice.git

WORKDIR /app/OpenVoice

# Install OpenVoice and its dependencies (including librosa==0.9.1)
RUN pip install --no-cache-dir -e .

# Install FastAPI, Uvicorn, and other API specific dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    librosa>=0.9.1 # Ensure a specific working version consistent with OpenVoice setup.py

# Create directories for checkpoints (though they will be mounted via docker-compose volumes)
RUN mkdir -p /app/checkpoints_v2/converter \
    && mkdir -p /app/checkpoints_v2/base_speakers/EN \
    && mkdir -p /app/checkpoints_v2/base_speakers/ses

RUN mkdir -p /app/huggingface_cache
ENV HF_HOME=/app/huggingface_cache

WORKDIR /app

# Copy your custom openvoice_api.py script into /app/
COPY ./Docker/openvoice_api.py /app/openvoice_api.py

EXPOSE 8000
CMD ["uvicorn", "openvoice_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]