#!/bin/bash

pushd $(dirname $0)
docker build -t plain-pytorch-benchmark .
docker run -it --rm --gpus all plain-pytorch-benchmark
