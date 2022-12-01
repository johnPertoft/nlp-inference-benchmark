import time

import torch
import torch._dynamo as torchdynamo
from kernl.model_optimization import optimize_model
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# TODO:
# - Larger models
# - Longer input sequences
# - More latency measurements
# - Do dynamic axes work? E.g. dynamic batch size and sequence length?
#   TensorRT could only handle one dynamic axis iirc?
# - Can we serialize the final optimized model easily?
# - Comment from kernl script: "# warmup (IRL, encoder and decoder should be warmed each on their own)"
# - We had some issues running the bf16 trained model with fp16 before, verify that all outputs
#   look ok.

NUM_MEASUREMENTS = 100
BATCH_SIZE = 1
MAX_LENGTH = 512

# default cache size needs to be increased to store the many graphs with generative models
torchdynamo.config.cache_size_limit = 512

model_name = "t5-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval().cuda()

inputs = tokenizer(
    ["translate English to German: The house in the woods is wonderful, can we buy it?"] * BATCH_SIZE,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    padding="max_length",
).to("cuda")

with torch.inference_mode():
    with torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        # Warmup.
        for _ in range(10):
            output = model.generate(
                inputs=inputs["input_ids"],
                min_length=22,
                max_length=22,
            )

        torch.cuda.synchronize()

        # Measure baseline latency.
        latency_baseline_total = 0.0
        for _ in range(NUM_MEASUREMENTS):
            start = time.perf_counter()
            output = model.generate(
                inputs=inputs["input_ids"],
                min_length=22,
                max_length=22,
            )
            latency_baseline_total += time.perf_counter() - start
        avg_latency_baseline = latency_baseline_total / NUM_MEASUREMENTS
        print(f"AVERAGE BASELINE LATENCY {avg_latency_baseline}")

# Optimize the model.
optimize_model(model.encoder)
optimize_model(model.decoder)

# This is the slow part.
print("RUNNING WARMUP (tracing?)")
with torch.inference_mode():
    with torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        start = time.perf_counter()
        model.generate(inputs=inputs["input_ids"], min_length=22, max_length=22)
        print(f"WARMUP TOOK {time.perf_counter() - start}")

with torch.inference_mode():
    with torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        # Warmup cuda/gpu.
        for _ in range(10):
            model.generate(
                inputs=inputs["input_ids"],
                min_length=22,
                max_length=22,
            )

        torch.cuda.synchronize()

        # Measure latencies.
        latency_total = 0.0
        for _ in range(NUM_MEASUREMENTS):
            start = time.perf_counter()
            output = model.generate(
                inputs=inputs["input_ids"],
                min_length=22,
                max_length=22,
            )
            latency_total += time.perf_counter() - start
        avg_latency_optimized = latency_total / NUM_MEASUREMENTS
        print(f"AVERAGE OPTIMIZED LATENCY {avg_latency_optimized}")
        print(f"{avg_latency_baseline / avg_latency_optimized:.1f}x speedup")
