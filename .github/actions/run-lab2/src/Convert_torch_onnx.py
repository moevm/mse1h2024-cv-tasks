import torch.onnx
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)

# Load the trained weights
model.load_state_dict(torch.load("./test_model.pth"))
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "test_model.onnx", verbose=True)


# import onnxruntime
# from torchvision import models
# import torch.nn as nn
# import torch
# import numpy as np

# # Load the ONNX model
# onnx_model_path = "./test_model.onnx"
# sess = onnxruntime.InferenceSession(onnx_model_path)

# # Get the names of the model's outputs
# output_names = [output.name for output in sess.get_outputs()]

# # Initialize the ResNet18 model without pre-trained weights
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 6)  # Set the number of output classes

# # Create a dummy input tensor with correct shape (batch_size, channels, height, width)
# dummy_input = torch.randn(1, 3, 224, 224).numpy()

# # Load the weights from the ONNX model into the ResNet18 model
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         if len(param.shape) == 4:
#             weights = np.transpose(sess.run([name], {'input.1': dummy_input})[0], (3, 2, 0, 1))  # Transpose the weights to match PyTorch format
#         else:
#             weights = sess.run([name], {'input.1': dummy_input})[0]
#         param.data = torch.from_numpy(weights)
#     elif 'bias' in name:
#         bias = sess.run([name], {'input.1': dummy_input})[0]
#         param.data = torch.from_numpy(bias)
