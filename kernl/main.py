import time

import torch
import torch._dynamo as torchdynamo
from kernl.model_optimization import optimize_model
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# default cache size needs to be increased to store the many graphs with generative models
torchdynamo.config.cache_size_limit = 512

# TODO: Change the model size when it's working
model_name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval().cuda()

# TODO: Try with longer input too
inputs = tokenizer(
    "translate English to German: The house in the woods is wonderful, can we buy it?",
    return_tensors="pt",
    pad_to_multiple_of=8,
    padding=True,
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

        # TODO: Why is this necessary?
        torch.cuda.synchronize()

        start = time.perf_counter()
        output = model.generate(
            inputs=inputs["input_ids"],
            min_length=22,
            max_length=22,
        )
        latency_baseline = time.perf_counter() - start
        print(latency_baseline)

        outputs = tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(outputs)

# Optimize the model.
optimize_model(model.encoder)
optimize_model(model.decoder)

# warmup (IRL, encoder and decoder should be warmed each on their own)
with torch.inference_mode():
    with torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        start = time.perf_counter()
        model.generate(inputs=inputs["input_ids"], min_length=22, max_length=22)
        print(time.perf_counter() - start)

with torch.inference_mode():
    with torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        # TODO: Why two different warmup?
        # Warmup again?
        for _ in range(10):
            model.generate(
                inputs=inputs["input_ids"],
                min_length=22,
                max_length=22,
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(
            inputs=inputs["input_ids"],
            min_length=22,
            max_length=22,
        )
        torch.cuda.synchronize()
        latency_optimized = time.perf_counter() - start
        print(latency_optimized)
        print(f"{latency_baseline/latency_optimized:.1f}x speedup")

        outputs = tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(outputs)
