FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    wget

# Kernl required python 3.9 specifically and the default
# apt repository has python 3.10.
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