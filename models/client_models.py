import torch
import torch.nn as nn
from torchvision import models
from typing import List

class MLPClientModel(nn.Module):

    def __init__(self, input_size: int, hidden_layers: List[int], embedding_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Final layer outputs embedding instead of prediction
        layers.append(nn.Linear(prev_size, embedding_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class ResNeXt29ClientModel(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        self.embedding_size = embedding_size

        # Load ResNeXt-29 (8x64d)
        self.resnext = models.resnext50_32x4d(pretrained=pretrained)  # Closest available in torchvision

        # Remove classification head and keep base features
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Identity()

        # Add embedding projection
        self.fc_embedding = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            embedding: Tensor of shape (batch_size, embedding_size)"
        """
        # Resize if needed (minimum 224x224 for pretrained weights)
        if x.shape[-2] < 224 or x.shape[-1] < 224:
            x = nn.functional.interpolate(
                x,
                size=(max(224, x.shape[-2]), max(224, x.shape[-1])),
                mode='bilinear',
                align_corners=False
            )

        # Extract features and project to embedding space
        features = self.resnext(x)
        return self.fc_embedding(features)
    
class ResNet50ClientModel(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        self.embedding_size = embedding_size

        # Load ResNeXt-29 (8x64d)
        self.resnext = models.resnet50(pretrained=pretrained)  # Closest available in torchvision

        # Remove classification head and keep base features
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Identity()

        # Add embedding projection
        self.fc_embedding = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            embedding: Tensor of shape (batch_size, embedding_size)"
        """
        # Resize if needed (minimum 224x224 for pretrained weights)
        if x.shape[-2] < 224 or x.shape[-1] < 224:
            x = nn.functional.interpolate(
                x,
                size=(max(224, x.shape[-2]), max(224, x.shape[-1])),
                mode='bilinear',
                align_corners=False
            )

        # Extract features and project to embedding space
        features = self.resnext(x)
        return self.fc_embedding(features)