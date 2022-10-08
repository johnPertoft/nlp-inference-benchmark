import time

import numpy as np
import onnxruntime as ort
import torch


session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

batch_size = 1
sequence_length = 512
input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int32).cpu().numpy()
inputs = {"input_ids": input_ids}

# Some warmup passes.
for _ in range(10):
    outputs = session.run(None, inputs)

timings = []
for _ in range(100):
    t1 = time.time()
    outputs = session.run(None, inputs)
    t2 = time.time()
    timings.append(t2 - t1)

avg_latency = np.mean(timings)
print(avg_latency)
