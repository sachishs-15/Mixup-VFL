import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from enum import Enum
from typing import List, Tuple, Dict, Optional, Callable
import copy
import tqdm
import pdb
from pdb import  set_trace
import argparse
from sklearn.datasets import fetch_california_housing
import random
import pandas as pd
import math
from typing import List, Optional
import wandb
from torchvision import models
import cv2
import xmltodict
import os
# from cocoData import load_coco_data
import torch.utils.checkpoint as cp
# from Loss import object_detection_loss
import torch.nn.functional as F
import datetime
class ClientModel(nn.Module):
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

class EfficientNetClientModel(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        self.embedding_size = embedding_size

        # Load pretrained EfficientNet-B3
        self.efficientnet = models.efficientnet_b3(pretrained=pretrained)

        # Remove classification head
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Add embedding projection
        self.fc_embedding = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            embedding: Tensor of shape (batch_size, embedding_size)
        """
        # Resize if needed (EfficientNet-B3 expects at least 300x300 input)
        if x.shape[-2] < 300 or x.shape[-1] < 300:
            x = nn.functional.interpolate(
                x,
                size=(max(300, x.shape[-2]), max(300, x.shape[-1])),
                mode='bilinear',
                align_corners=False
            )

        # Extract features and project to embedding space
        features = self.efficientnet(x)
        return self.fc_embedding(features)



class ServerModel(nn.Module):
    def __init__(self, num_clients: int, embedding_size: int, hidden_layers: List[int], output_size: int = 1):
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

        layers.append(nn.Linear(prev_size, output_size))
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
       #  ()
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