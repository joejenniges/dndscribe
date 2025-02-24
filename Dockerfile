# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies including opus and ffmpeg
RUN apt-get update && apt-get install -y \
    libopus0 \
    libopus-dev \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p recordings && mkdir -p models

# Set environment variables for better PyTorch performance
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4

# Copy the .env file
COPY .env .

# We'll mount bot.py at runtime instead of copying it here
CMD ["python", "bot.py"]