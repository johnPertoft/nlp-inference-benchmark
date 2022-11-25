#!/bin/bash

pushd $(dirname $0)
DOCKER_BUILDKIT=0 docker build -t onnx-runtime-tvm .
docker run -it --rm --gpus all onnx-runtime-tvm
