# Model2 with 1X1 convs following usual Convs

## Number of parameters:2.8k

### Best train accuracy:93.26

### Best test accuracy: 97.9

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
            nn.Conv2d(1, 8, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # 1X1 convolutions
        self.conv2_pointwise = nn.Sequential(
            nn.Conv2d(16, 8, 1, padding=0),
             nn.ReLU(),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv3_pointwise = nn.Sequential(
            nn.Conv2d(16, 8, 1, padding=0),
             nn.ReLU(),
            )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(64, 10, 3, padding=1),
        #    nn.ReLU(),
            #nn.BatchNorm2d(16),
            #nn.MaxPool2d(2, 2),
        #)

        self.linearLayer = nn.Linear(8,10)
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0),
        )

        # self.output_size = self._get_conv_output_size((1, 28, 28))
        #self.fc1 = nn.Sequential(
        #    nn.Linear(self.output_size , 32),
        #)
        self.fc2 = nn.Sequential(
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2_pointwise(x)
        x = self.conv3(x)
        x = self.conv3_pointwise(x)
        x = self.gap(x)
        #x = self.conv4(x)
        x = x.view(-1,8)
        x = self.linearLayer(x)
        #x = self.fc1(x)
        #x = self.fc2(x)
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

