import onnx
import onnxsim
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
    export_path = "model.onnx"

    torch.onnx.export(
        model,
        input_ids,
        export_path,
        input_names=["input_ids"],
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    model_onnx = onnx.load(export_path)
    onnx.checker.check_model(model_onnx)

    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "onnx simplify failed"
    onnx.save(model_onnx, export_path)
