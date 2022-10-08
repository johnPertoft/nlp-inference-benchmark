FROM nvidia/cuda:11.6.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libssl-dev \
    pkg-config \
    unzip
ENV PATH=/usr/local/nvidia/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ARG LIBTORCH_URL="https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip"
ENV LIBTORCH="/usr/lib/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"
RUN mkdir -p $LIBTORCH && \
    curl $LIBTORCH_URL -o libtorch.zip && \
    unzip -u libtorch.zip -d $(dirname "${LIBTORCH}") && \
    rm libtorch.zip
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.cargo/bin:$PATH
COPY Cargo.toml Cargo.toml
COPY src src
ENTRYPOINT [ "cargo", "run", "--release" ]