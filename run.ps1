# Set resource limits
$MEMORY = "8g"
$CPUS = "4"
$SHM_SIZE = "2g"  # Increased shared memory for better performance

# Create temp directory if it doesn't exist
$tempDir = Join-Path $PWD "temp"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
    Write-Host "Created temp directory: $tempDir"
}

# Stop any existing container
Write-Host "Stopping any existing dndscribe-bot container..."
docker stop dndscribe-bot 2>$null

# Run the container with optimized settings
Write-Host "Starting dndscribe-bot container with improved network settings..."
docker run `
    --memory=$MEMORY `
    --memory-swap=$MEMORY `
    --cpus=$CPUS `
    --shm-size=$SHM_SIZE `
    -v ${PWD}/src:/app/src `
    -v ${PWD}/ignored.txt:/app/ignored.txt `
    -v ${PWD}/.env:/app/.env `
    -v ${PWD}/recordings:/app/recordings `
    -v ${PWD}/models:/app/models `
    -v ${PWD}/logs:/app/logs `
    -v ${tempDir}:/tmp/audio_processing `
    --name dndscribe-bot `
    dndscribe

Write-Host "Container has stopped. Check logs for details."