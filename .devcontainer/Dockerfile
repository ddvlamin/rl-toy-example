# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3
FROM mcr.microsoft.com/vscode/devcontainers/python:3.12

# Copy project file
COPY . /workspace
WORKDIR /workspace

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
USER root

# [Optional] Add sudo support for the non-root user
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode

# [Optional] Set the default user. Omit if you want to connect as root.
USER vscode



