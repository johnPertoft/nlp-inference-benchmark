docker build -t plain-pytorch-benchmark .
docker run -it --rm --gpus all plain-pytorch-benchmark
