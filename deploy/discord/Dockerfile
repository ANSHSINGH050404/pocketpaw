# Full-featured Discord bot with all agent backends, browser, and tools

# ---- Node.js stage ----
FROM node:22.14.0-slim AS node

RUN npm install -g @anthropic-ai/claude-code @openai/codex && \
    npm cache clean --force

# ---- Builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir '.[all]'

# Install Playwright Chromium browser
RUN playwright install chromium

# ---- Runtime stage ----
FROM python:3.12-slim

# Runtime system deps: git, tesseract for OCR, curl for healthcheck,
# shared libs required by Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tesseract-ocr \
    curl \
    # Playwright Chromium shared libs
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    libx11-xcb1 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js + globally-installed CLI backends from the official node image
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js /usr/local/bin/claude && \
    ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Playwright browsers from builder
COPY --from=builder /root/.cache/ms-playwright /home/pocketpaw/.cache/ms-playwright

# Create non-root user
RUN groupadd --system pocketpaw && \
    useradd --system --gid pocketpaw --create-home pocketpaw && \
    mkdir -p /home/pocketpaw/.pocketpaw /home/pocketpaw/workspace && \
    chown -R pocketpaw:pocketpaw /home/pocketpaw

USER pocketpaw
WORKDIR /home/pocketpaw

# Agent-created files land here — bind-mount to access them on the host
ENV POCKETPAW_FILE_JAIL_PATH=/home/pocketpaw/workspace

CMD ["pocketpaw", "--discord"]
