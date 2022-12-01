FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt-get update --fix-missing
RUN apt install -y \
    git \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    python-is-python3
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN pip install \
    jinja2 \
    numpy
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install AITemplate
RUN git clone --recursive https://github.com/facebookincubator/AITemplate
RUN cd AITemplate/python && python setup.py bdist_wheel && pip install dist/*.whl --force-reinstall