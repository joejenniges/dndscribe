#!/bin/bash
# run.sh

# Set the memory limit (e.g., 8GB)
MEMORY="8g"

# Set CPU limits (e.g., use up to 4 CPUs)
CPUS="4"

# Run the container with resource limits
docker run \
    --memory=$MEMORY \
    --memory-swap=$MEMORY \
    --cpus=$CPUS \
    --shm-size=1g \
    -v $(pwd)/bot.py:/app/bot.py \
    -v $(pwd)/recordings:/app/recordings \
    -v $(pwd)/models:/app/models \
    dndscribe