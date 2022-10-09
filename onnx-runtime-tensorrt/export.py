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

"""
torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
    training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
    do_constant_folding=not train,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                } if dynamic else None)


model_onnx, check = onnxsim.simplify(
    model_onnx,
    dynamic_input_shape=dynamic,
    input_shapes={'images': list(im.shape)} if dynamic else None)
"""