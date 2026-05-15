FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome-stable

RUN apt-get update && apt-get install curl gnupg -y \
    && curl --location --silent https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
    && apt-get update \
    && apt-get install google-chrome-stable -y --no-install-recommends

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
    less \
    git \
    procps \
    fzf \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
    jq \
    nano \
    vim \
    python3 \
    python3-pip \
    ca-certificates \
    ffmpeg \
    python3-dev \
    locales \
    mesa-utils \
    libgbm1 \
    libegl1 \
    libwayland-client0 \
    dbus \
    pulseaudio-utils \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uvx /usr/local/bin/uvx

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R node:node /usr/local/share

ARG USERNAME=node

# Persist bash history.
RUN mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace/ophir /home/node/.claude \
    /home/node/.cache/uv \
    /home/node/.local/share \
    /home/node/.config/google-chrome \
    /home/node/.config/matplotlib \
    /etc/opt/chrome/policies/managed \
    /etc/opt/chrome/policies/recommended \
    /home/node/.npm \
    /run/user/1000 && \
    chown -R node:node /workspace /home/node/.claude \
    /home/node/.cache /home/node/.local/share \
    /home/node/.config/google-chrome \
    /home/node/.config/matplotlib \
    /home/node/.npm && \
    chown -R node:node /etc/opt/chrome/policies && \
    chown node:node /run/user/1000 && chmod 700 /run/user/1000

WORKDIR /workspace/ophir

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Set up non-root user
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV NODE_PATH=/usr/local/share/npm-global/lib/node_modules

# Set the default shell to bash
ENV SHELL=/bin/bash

# Set the default editor and visual
ENV EDITOR=nano
ENV VISUAL=nano

# Install oh-my-bash
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" --unattended && \
    echo '[ -f /usr/share/doc/fzf/examples/key-bindings.bash ] && source /usr/share/doc/fzf/examples/key-bindings.bash' >> ~/.bashrc && \
    echo '[ -f /usr/share/doc/fzf/examples/completion.bash ] && source /usr/share/doc/fzf/examples/completion.bash' >> ~/.bashrc && \
    echo 'alias python="python3"' >> ~/.bashrc && \
    echo 'export HISTFILE=/commandhistory/.bash_history' >> ~/.bashrc && \
    echo 'export PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND; }history -a"' >> ~/.bashrc


# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Install Puppeteer (uses system Chrome) and its MCP server for Claude Code
RUN npm install -g puppeteer @modelcontextprotocol/server-puppeteer

# Copy and set up firewall and search scripts
COPY init-firewall.sh /usr/local/bin/
COPY search.sh /usr/local/bin/search
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh /usr/local/bin/search

USER node

# Add launch-chrome helper (native Wayland + NVIDIA ANGLE/OpenGL GPU acceleration)
RUN { \
    echo ''; \
    echo 'launch-chrome() {'; \
    echo '    dbus-run-session -- google-chrome-stable \'; \
    echo '        --ozone-platform=wayland \'; \
    echo '        --enable-features=UseOzonePlatform,WaylandWindowDecorations \'; \
    echo '        --use-gl=angle \'; \
    echo '        --use-angle=opengl \'; \
    echo '        --enable-gpu-rasterization \'; \
    echo '        --enable-zero-copy \'; \
    echo '        --ignore-gpu-blocklist \'; \
    echo '        --disable-dev-shm-usage \'; \
    echo '        --no-sandbox \'; \
    echo '        --disable-setuid-sandbox \'; \
    echo '        "$@" &'; \
    echo '}'; \
    } >> ~/.bashrc
