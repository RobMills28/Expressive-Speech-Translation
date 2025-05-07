FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with specific version
RUN pip install torch==1.13.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

# Clone OpenVoice repository
WORKDIR /app
RUN git clone https://github.com/myshell-ai/OpenVoice.git

# Install OpenVoice in development mode
WORKDIR /app/OpenVoice
RUN pip install -e .

# Install dependencies for the API
RUN pip install fastapi uvicorn pydantic python-multipart

# Create checkpoint directories
RUN mkdir -p /app/checkpoints_v2/converter \
    && mkdir -p /app/checkpoints_v2/base_speakers/EN \
    && mkdir -p /app/checkpoints_v2/base_speakers/ZH \
    && mkdir -p /app/checkpoints_v2/base_speakers/ses \
    && mkdir -p /app/processed

# Return to app directory
WORKDIR /app

# Copy our API wrapper code
COPY ./Docker/openvoice_api.py /app/

# Expose the port
EXPOSE 8000

# Set Python path to ensure module can be found
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run the API server with Python directly
CMD ["python", "-c", "import uvicorn; uvicorn.run('openvoice_api:app', host='0.0.0.0', port=8000)"]