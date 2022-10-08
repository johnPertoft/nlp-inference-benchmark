#!/bin/bash

pushd $(dirname $0)
docker build -t pytorch-torchscript .
docker run -it --rm --gpus all pytorch-torchscript