#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
#FROM python:3.9
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    wget
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN pip install git+https://github.com/ELS-RD/kernl --extra-index-url https://download.pytorch.org/whl/nightly/cu117
RUN pip install \
    sentencepiece \
    transformers
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]