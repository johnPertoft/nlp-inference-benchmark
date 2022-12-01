#!/bin/bash
set -e

pushd $(dirname $0)
docker build -t kernl .
docker run -it --rm --gpus all kernl
