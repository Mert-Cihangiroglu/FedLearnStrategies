# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def get_model(output_classes, input_channels):
    """
    Dynamically create and adjust ResNet18 based on input channels and output classes.

    Parameters:
        output_classes (int): Number of output classes for the dataset.
        input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).

    Returns:
        torch.nn.Module: Modified ResNet18 model.
    """
    # Load ResNet18
    model = resnet18(weights=None)
    
    # Adjust the first convolutional layer for input channels
    if input_channels != 3:
        model.conv1 = torch.nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    
    # Adjust the final fully connected layer for output classes
    model.fc = torch.nn.Linear(model.fc.in_features, output_classes)
    
    return model

# Example Usage
if __name__ == "__main__":
    num_classes = 10  # Example for CIFAR-10
    model = get_model(output_classes=num_classes)
    print(model)
