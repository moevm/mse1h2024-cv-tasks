# import onnx
# import onnxruntime
# from torchvision import models
# import torch.nn as nn
# import torch

# # Load the ONNX model
# onnx_model = onnx.load("./test_model.onnx")

# # Convert the ONNX model to a PyTorch model
# ort_session = onnxruntime.InferenceSession("./test_model.onnx")
# input_names = [input.name for input in ort_session.get_inputs()]
# output_names = [output.name for output in ort_session.get_outputs()]

# # Define the ResNet18 model without pre-trained weights
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 6)

# # Optionally, move the model to CUDA if available
# if torch.cuda.is_available():
#     model = model.cuda()

###################################################### v2
import torch
import torchvision.models as models
import onnx

# Loading the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Setting the model to evaluation mode
model.eval()

# # Defining the input tensor for the model
# x = torch.randn(1, 3, 224, 224, requires_grad=True)

# # Exporting the model to ONNX format
# torch.onnx.export(model,  # model
#                   x,  # input tensor
#                   "resnet18.onnx",  # file name
#                   export_params=True,  # save weights and biases
#                   opset_version=10,  # ONNX version
#                   do_constant_folding=True)  # constant folding optimization

# Loading the ONNX model
onnx_model = onnx.load("resnet18.onnx")
