#!/bin/bash
set -e

RUN_DIR="$1"
if [ ! -d "$RUN_DIR" ]; then
    echo "$RUN_DIR is not a directory"
    exit 1
fi

pushd "$RUN_DIR"
docker build -t "$RUN_DIR" .
mkdir -p /tmp/.cache
docker run -it --rm -v /tmp/.cache:/root/.cache --gpus all "$RUN_DIR"
