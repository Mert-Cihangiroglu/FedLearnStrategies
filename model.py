# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def get_model(output_classes):
    """
    Returns a ResNet18 model customized for the specified number of output classes.

    Parameters:
        output_classes (int): Number of output classes for the dataset.

    Returns:
        model (torch.nn.Module): Customized ResNet18 model.
    """
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, output_classes)
    return model

# Example Usage
if __name__ == "__main__":
    num_classes = 10  # Example for CIFAR-10
    model = get_model(output_classes=num_classes)
    print(model)
