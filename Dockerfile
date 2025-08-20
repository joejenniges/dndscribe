# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies including opus and ffmpeg with better audio support
RUN apt-get update && apt-get install -y \
    libopus0 \
    libopus-dev \
    ffmpeg \
    git \
    build-essential \
    libnuma-dev \
    libsoxr-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p recordings && \
    mkdir -p models && \
    mkdir -p /tmp/audio_processing && \
    mkdir -p logs

# Set environment variables for better performance
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TEMP=/tmp/audio_processing \
    TMPDIR=/tmp/audio_processing

# Set network optimizations for Discord voice connections
RUN echo "net.core.rmem_max=2097152" >> /etc/sysctl.conf && \
    echo "net.core.wmem_max=2097152" >> /etc/sysctl.conf && \
    echo "net.core.rmem_default=1048576" >> /etc/sysctl.conf && \
    echo "net.core.wmem_default=1048576" >> /etc/sysctl.conf

# FFmpeg optimizations - create a basic config
RUN mkdir -p /root/.ffmpeg && \
    echo "[generic]" > /root/.ffmpeg/ffmpeg.config && \
    echo "thread_queue_size=4096" >> /root/.ffmpeg/ffmpeg.config

# Copy the .env file
COPY .env .

# Configure Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import socket; socket.socket().connect(('discord.com', 443))" || exit 1

# Set the entrypoint
ENTRYPOINT ["python", "src/bot.py"]