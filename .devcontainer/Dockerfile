FROM ubuntu:20.04

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    git \
    htop \
    jq \
    locales \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    ripgrep \
    sudo \
    tmux \
    vim \
    wget \
    zsh \
    && apt-get clean \
    && rm --recursive --force /var/lib/apt/lists/*

# Install python dev dependencies in separate environments via
# pipx to avoid dependency conflicts.
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=$PATH:$PIPX_BIN_DIR
RUN pip install pipx && pipx ensurepath
RUN pipx install black==22.3.0 \
    && pipx install dvc[gs]==2.10.1 \
    && pipx install flake8==4.0.1 \
    && pipx install isort==5.10.1 \
    && pipx install mypy==0.942 \
    && pipx install pre-commit==2.18.1 \
    && pipx install pytest==7.1.1

# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN \
    groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/zsh $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user.
USER $USERNAME

# Install Oh-my-zsh.
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
