import torch
from transformers import BertConfig
from transformers import BertModel


# TODO:
# - Define the opset when exporting
# - Use dynamic shapes?
# - Any other things to enable? constand folding etc?

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
model.eval()

batch_size = 1
sequence_length = 512
input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int32)

with torch.inference_mode():
    torch.onnx.export(model, input_ids, "model.onnx", input_names=["input_ids"])
