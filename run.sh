#!/bin/bash
# Build and run dndscribe-ts
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t dndscribe-ts "$SCRIPT_DIR"

docker run -d \
    --name dndscribe-ts \
    --restart unless-stopped \
    --env-file "$SCRIPT_DIR/.env" \
    -v "$SCRIPT_DIR/recordings:/app/recordings" \
    -v "$SCRIPT_DIR/models:/app/models" \
    -v "$SCRIPT_DIR/logs:/app/logs" \
    -v "$SCRIPT_DIR/ignored.txt:/app/ignored.txt" \
    dndscribe-ts

echo "Container started. Logs: docker logs -f dndscribe-ts"
