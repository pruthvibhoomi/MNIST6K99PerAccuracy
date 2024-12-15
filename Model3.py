# Model 3 .

# Now I try to extract more features and output channels

# Final Result of Model 3
### Number of parameters: 10.8k

### Best train accuracy: 96.5

### Best test accuracy: 98.9

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # 1X1 convolutions
        self.conv2_pointwise = nn.Sequential(
            nn.Conv2d(32, 16, 1, padding=0),
             nn.ReLU(),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv3_pointwise = nn.Sequential(
            nn.Conv2d(32, 16, 1, padding=0),
             nn.ReLU(),
            )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linearLayer = nn.Linear(16,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2_pointwise(x)
        x = self.conv3(x)
        x = self.conv3_pointwise(x)
        x = self.gap(x)
        x = x.view(-1,16)
        x = self.linearLayer(x)
        return F.log_softmax(x,dim=1)

    def _get_conv_output_size(self, input_size):
        """
        Calculates the output size of the convolutional layers for a given input size.
        This is used to dynamically determine the input size for the first fully connected layer.
        """
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_size)
            # Pass the dummy input through the convolutional layers
            output = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
            # Calculate the flattened output size
            output_size = output.numel() // output.size(0)
            return output.shape[1] * output.shape[2] * output.shape[3]

