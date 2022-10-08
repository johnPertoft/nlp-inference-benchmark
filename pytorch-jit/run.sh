#!/bin/bash

pushd $(dirname $0)
docker build -t pytorch-jit .
docker run -it --rm --gpus all pytorch-jit