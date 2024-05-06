import onnx
import onnxruntime
from torchvision import models
import torch.nn as nn
import torch

# Load the ONNX model
onnx_model = onnx.load("./test_model.onnx")

# Convert the ONNX model to a PyTorch model
ort_session = onnxruntime.InferenceSession("./test_model.onnx")
input_names = [input.name for input in ort_session.get_inputs()]
output_names = [output.name for output in ort_session.get_outputs()]

# Define the ResNet18 model without pre-trained weights
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)

# Optionally, move the model to CUDA if available
if torch.cuda.is_available():
    model = model.cuda()
