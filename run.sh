#!/bin/bash
# run.sh

# Set the memory limit (e.g., 8GB)
MEMORY="8g"

# Set CPU limits (e.g., use up to 4 CPUs)
CPUS="4"

# Run the container with resource limits
docker run \
    --rm \
    --memory=$MEMORY \
    --memory-swap=$MEMORY \
    --cpus=$CPUS \
    --shm-size=2g \
    -v ${PWD}/src:/app/src \
    -v ${PWD}/ignored.txt:/app/ignored.txt \
    -v ${PWD}/.env:/app/.env \
    -v ${PWD}/recordings:/app/recordings \
    -v ${PWD}/models:/app/models \
    --name dndscribe-bot \
    dndscribe 