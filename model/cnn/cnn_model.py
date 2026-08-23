"""
TrafficCNN Model

This module defines the Convolutional Neural Network (CNN) used for
vehicle classification in the AI Traffic Detection System.

Algorithm:
- MobileNetV2 (Transfer Learning)

Classes:
- Car
- Truck
- Bus
- Motorcycle
- Bicycle
"""

import torch
import torch.nn as nn
from torchvision import models


class TrafficCNN(nn.Module):

    def __init__(self, num_classes=5, pretrained=True):
        super(TrafficCNN, self).__init__()

        weights = (
            models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.model = models.mobilenet_v2(
            weights=weights
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
