# Inference benchmarks/playground
This repo holds benchmarks for some deep learning inference frameworks for our specific workload.


## Results
TODO

## TODO
- Define some delimitations, e.g. only running on 3080, only using T5 etc
- Define some common output report format
- Don't use Bert here
- Plain pytorch/transformer model, also compare different implementations?
- Try with FlashAttention module?
- Try with TorchScript
- Try with torch.jit.trace
- Try lower precisions etc?
- See https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
- Pin dependency versions
- Share huggingface cache somewhere?
- What's the best way to time things?
- Run model() or model.forward()?
- Check both latency and throughput?
- Is it even possible to trace the .generate(..) method as well? Or do we need
  to manually call the forward pass multiple times?
- For the onnx models, is it a problem not having access to gpu when exporting?
  (since we run the export during docker build)
- Be consistent with cuda/cudnn versions etc
- Should verify correctness in output?
- For onnx, we need to use a tool to perform optimizations of the graph I think.
- https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/notebooks/t5.ipynb seems relevant
