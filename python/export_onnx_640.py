import torch
from train.train_student_sr import StudentSR

model = StudentSR(num_blocks=8, ch=64)
ckpt = torch.load("../weights/student_sr.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy = torch.randn(1, 3, 480, 640)

torch.onnx.export(
    model,
    dummy,
    "../weights/student_sr_640x480.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes=None,    # ВАЖНО: model becomes static with fixed shape
)
