# WHY node:18 not node:20+: dysnomia targets Node 18, matching Craig's setup.
# WHY not alpine: sodium-native and @discordjs/opus need native compilation,
# and alpine's musl libc causes issues with some native modules.
FROM node:18-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    build-essential \
    libopus0 libopus-dev \
    libsodium23 libsodium-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install whisper in a venv
RUN python3 -m venv /opt/whisper-venv \
    && /opt/whisper-venv/bin/pip install --no-cache-dir openai-whisper
ENV PATH="/opt/whisper-venv/bin:$PATH"

WORKDIR /app

# Install node dependencies first (layer caching)
COPY package.json ./
RUN npm install

# Copy source and build
COPY tsconfig.json ./
COPY src/ src/
RUN npm run build

# Create directories
RUN mkdir -p recordings logs models

ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD node -e "try{require('net').createConnection({port:0});process.exit(0)}catch(e){process.exit(1)}"

ENTRYPOINT ["node", "dist/bot.js"]
