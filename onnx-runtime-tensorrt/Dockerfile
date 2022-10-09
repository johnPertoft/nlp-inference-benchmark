# TODO: Should use 515 driver with this?
# TODO: Or earlier container maybe
FROM nvcr.io/nvidia/tensorrt:22.09-py3
RUN pip install \
    onnx \
    onnxruntime-gpu \
    torch \
    transformers
COPY export.py export.py
RUN python export.py
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]