FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-get update && apt-get install -y \
    ca-certificates \
    g++ \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3
RUN pip install \
    onnx \
    onnxruntime-gpu \
    torch \
    transformers
COPY export.py export.py
RUN python export.py
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]

# RUN apt-get update && apt-get install -y --no-install-recommends python3-dev ca-certificates g++ python3-numpy gcc make git python3-setuptools python3-wheel python3-pip aria2 && aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr
# RUN cd /code && /bin/bash ./build.sh --skip_submodule_sync --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_cuda --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) 'CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;86'
# FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
