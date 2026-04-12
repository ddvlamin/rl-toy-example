# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project file
COPY . /workspace
WORKDIR /workspace

#COPY pyproject.toml uv.lock ./
#COPY src ./src

# Install dependencies using uv
RUN uv sync --frozen

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
USER root

# [Optional] Add sudo support for the non-root user
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends npm libsecret-1-0 sudo \
    && npm install -g @google/gemini-cli \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode

# [Optional] Set the default user. Omit if you want to connect as root.
USER vscode

RUN echo 'git config --global user.email "ddvlamin@gmail.com"' >> ~/.bashrc && \
    echo 'git config --global user.name "Dieter Devlaminck"' >> ~/.bashrc
