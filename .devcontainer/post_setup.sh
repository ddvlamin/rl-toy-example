#!/bin/bash
#if ! grep -q "GITHUB_TOKEN" ~/.bashrc; then
#    echo 'export GITHUB_TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential fill 2>&1 | grep "^password=" | cut -d= -f2)' >> ~/.bashrc
#fi

npm install -g @modelcontextprotocol/server-filesystem @modelcontextprotocol/server-github @modelcontextprotocol/server-sequential-thinking

uv sync --group dev --frozen