import torch
import torchvision.models as models
import onnx

# Loading the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Setting the model to evaluation mode
model.eval()

# Defining the input tensor for the model
x = torch.randn(1, 3, 224, 224, requires_grad=True)

# Exporting the model to ONNX format
torch.onnx.export(model,  # model
                  x,  # input tensor
                  "resnet18.onnx",  # file name
                  export_params=True,  # save weights and biases
                  opset_version=10,  # ONNX version
                  do_constant_folding=True)  # constant folding optimization

# Loading the ONNX model
onnx_model = onnx.load("resnet18.onnx")
