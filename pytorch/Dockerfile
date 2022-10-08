FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
RUN pip install \
    transformers
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]
