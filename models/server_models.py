import torch
import torch.nn as nn
from torchvision import models
from typing import List

class ServerModel_Regression(nn.Module):
    def __init__(self, num_clients: int, embedding_size: int, hidden_layers: List[int]):
        super().__init__()
        
        total_input_size = num_clients * embedding_size

        layers = []
        prev_size = total_input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, embeddings: List[torch.Tensor]):
        # Concatenate embeddings from all clients
        combined = torch.cat(embeddings, dim=1)

        return self.layers(combined)

class ResNetBBoxPredictor(nn.Module):

    def __init__(self, num_clients, total_embedding_size: int, num_classes: int, num_boxes: int = 5, conf_threshold: float = 0.7):
        """
        Args:
            total_embedding_size: Input embedding size from ResNet
            num_classes: Number of object classes
            num_boxes: Maximum bounding boxes per image
            conf_threshold: Confidence threshold for filtering predictions
        """
        super().__init__()
        self.num_boxes = num_boxes
        self.conf_threshold = conf_threshold

        self.fc1 = nn.Linear(total_embedding_size, 512)
        self.relu1 = nn.ReLU()
        self.bbox_head = nn.Linear(512, num_boxes * 4)
        self.class_head = nn.Linear(512, num_boxes * num_classes)
        self.conf_head = nn.Linear(512, num_boxes)

    def forward(self, x):

        # Apply intermediate layers
        x = self.fc1(x)
        x = self.relu1(x)

        bbox_preds = self.bbox_head(x)
        class_preds = self.class_head(x)  # (batch, num_boxes * num_classes)
        conf_preds = torch.sigmoid(self.conf_head(x))  # (batch, num_boxes)

        # Reshape outputs
        bbox_preds = bbox_preds.view(-1, self.num_boxes, 4)  # (batch, num_boxes, 4)
        class_preds = class_preds.view(-1, self.num_boxes, 91)  # (batch, num_boxes, num_classes)
        conf_preds = conf_preds.view(-1, self.num_boxes, 1)  # (batch, num_boxes, 1)
        return bbox_preds,class_preds,conf_preds
    