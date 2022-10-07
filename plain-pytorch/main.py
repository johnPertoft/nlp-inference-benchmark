import time

import numpy as np
import torch
from transformers import BertConfig
from transformers import BertModel

# TODO: Try several approaches here
# - Don't use Bert here
# - Plain pytorch/transformer model, also compare different implementations?
# - Try with FlashAttention module?
# - Try with TorchScript
# - Try with torch.jit.trace
# - Try lower precisions etc?
# - See https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
# - Pin dependency versions
# - Share huggingface cache somewhere?
# - What's the best way to time things?
# - Run model() or model.forward()?

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)

batch_size = 1
sequence_length = 512
input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int32)

model = model.to("cuda")
input_ids = input_ids.to("cuda")

# Warmup relevant here?
for _ in range(10):
    model.forward(input_ids)

timings = []
for _ in range(10):
    t1 = time.time()
    model.forward(input_ids)
    t2 = time.time()
    timings.append(t2 - t1)

avg_latency = np.mean(timings)
print(avg_latency)
