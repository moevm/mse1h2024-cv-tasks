from torchvision import models
import torch.nn as nn
import torch

model_file = "./models/test_model.pth"

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    state_dict = torch.load(model_file)  # Load the model's state dictionary
else:
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))

# Initialize the ResNet18 model without pre-trained weights
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)

# Load the trained weights into the model
model.load_state_dict(state_dict)
