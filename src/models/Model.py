from torchvision import models
import torch.nn as nn
import torch
model_file = "./models/test_model.pth"

if torch.cuda.is_available():
    state_dict = torch.load(model_file)
else:
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))

model = models.resnet18(pretrained=False)  # Initialize the ResNet18 model without pre-trained weights
model.fc = nn.Linear(model.fc.in_features, 6)  # Assuming the last layer has 6 output features
model.load_state_dict(state_dict)