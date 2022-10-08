import time

import numpy as np
import torch
from transformers import BertConfig
from transformers import BertModel


config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
model.eval()

batch_size = 1
sequence_length = 512
input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int32)

model = model.to("cuda")
input_ids = input_ids.to("cuda")

with torch.inference_mode():
    # Some warmup passes.
    for _ in range(10):
        model.forward(input_ids)

    timings = []
    for _ in range(1000):
        t1 = time.time()
        model.forward(input_ids)
        t2 = time.time()
        timings.append(t2 - t1)

    avg_latency = np.mean(timings)
    print(avg_latency)
