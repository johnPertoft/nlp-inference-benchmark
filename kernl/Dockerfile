#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
FROM python:3.9
RUN apt-get update && apt-get install -y \
    git
RUN pip install git+https://github.com/ELS-RD/kernl --extra-index-url https://download.pytorch.org/whl/nightly/cu117
RUN pip install \
    sentencepiece \
    transformers
COPY main.py main.py
ENTRYPOINT [ "python", "main.py" ]