#FROM tlcpack/ci-gpu:v0.55
FROM tlcpack/ci-gpu:20220925-060158-71f25b3d6

RUN apt-get update && apt-get install -y \
    llvm-16 \
    llvm-16-runtime \
    llvm-16-dev
RUN apt --fix-broken install
COPY install_tvm_gpu.sh /install/install_tvm_gpu.sh
RUN bash /install/install_tvm_gpu.sh

ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/vta/python:${PYTHONPATH}
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


RUN pip install \
    onnx \
    onnxruntime-gpu \
    onnxsim \
    torch \
    transformers
COPY export.py export.py
RUN python export.py
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]


# FROM tlcpack/package-cu116:b62a9ce
# RUN pip install \
#     onnx \
#     onnxruntime-gpu \
#     onnxsim \
#     torch \
#     transformers
# #RUN pip install tlcpack-nightly-cu102 -f https://tlcpack.ai/wheels
# COPY export.py export.py
# RUN python export.py
# COPY main.py main.py
# ENTRYPOINT [ "python", "main.py" ]