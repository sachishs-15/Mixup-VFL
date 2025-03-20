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
from pdb import set_trace
import argparse
from sklearn.datasets import fetch_california_housing
import random
import pandas as pd
import math
from typing import List, Optional
from torchvision import models
import cv2
import xmltodict
import os
from coco8Data import load_coco8_data
from Loss import object_detection_loss
import torch.nn.functional as F

class DataAlignment(Enum):
    ALIGNED = "aligned"
    UNALIGNED = "unaligned"

class MixupStrategy(Enum):
    NO_MIXUP = "no_mixup"
    MAX_MIXUP = "max_mixup"
    MEAN_MIXUP = "mean_mixup"
    IMPORTANCE_MIXUP = "importance_mixup"
    MODEL_BASED_MIXUP = "model_based_mixup"
    MUTUAL_INFO_MIXUP = "mutual_info_mixup"

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

class ResNet50ClientModel(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, image_width: Optional[int] = None, pretrained: bool = True):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        
        # Define image width
        if image_width is None:
            self.image_width = int(math.sqrt(input_size // 3))  # Adjust for 3 channels
        else:
            self.image_width = image_width
            
        # Compute image height based on input size
        self.image_height = math.ceil(input_size / (self.image_width * 3))  # 3 channels

        # Load a pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the first convolution layer to accept 3-channel RGB input (default is already 3)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Final embedding layer
        self.fc_embedding = nn.Linear(num_features, embedding_size)
    
    def forward(self, x):
        # batch_size = x.shape[0]
        
        # # Compute required padding
        # padded_size = self.image_width * self.image_height * 3  # Adjust for 3 channels
        # if self.input_size < padded_size:
        #     padding = torch.zeros(batch_size, padded_size - self.input_size, device=x.device)
        #     print(padding.shape)
        #     print(x.shape)
        #     x = torch.cat([x, padding], dim=1)
        
        # # Reshape to (batch_size, 3, H, W)
        # x = x.view(batch_size, 3, self.image_height, self.image_width)

        # Ensure input is large enough for ResNet (at least 224x224)
        if self.image_height < 224 or self.image_width < 224:
            x = nn.functional.interpolate(
                x, 
                size=(max(224, self.image_height), max(224, self.image_width)),
                mode='bilinear',
                align_corners=False
            )

        x = self.resnet(x)
        embedding = self.fc_embedding(x)
        
        return embedding
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
        self.conf_threshold = conf_threshold  # Minimum confidence to keep a box
        
        # Add intermediate layers to go from total_embedding_size to 512
        # set_trace()
        self.fc1 = nn.Linear(total_embedding_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        
        # Bounding box prediction: (x_min, y_min, x_max, y_max) per box
        self.bbox_head = nn.Linear(512, num_boxes * 4)
        # Class probabilities per box
        self.class_head = nn.Linear(512, num_boxes * num_classes)
        # Objectness score per box
        self.conf_head = nn.Linear(512, num_boxes)
        
    def forward(self, x):
        # Apply intermediate layers
       # set_trace()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        bbox_preds = self.bbox_head(x)  # (batch, num_boxes * 4)
        class_preds = self.class_head(x)  # (batch, num_boxes * num_classes)
        conf_preds = torch.sigmoid(self.conf_head(x))  # (batch, num_boxes)
        
        # Reshape outputs
        bbox_preds = bbox_preds.view(-1, self.num_boxes, 4)  # (batch, num_boxes, 4)
        class_preds = class_preds.view(-1, self.num_boxes, 80)  # (batch, num_boxes, num_classes)
        conf_preds = conf_preds.view(-1, self.num_boxes, 1)  # (batch, num_boxes, 1)
        
        # Apply confidence threshold: Keep only boxes with conf > self.conf_threshold
        return bbox_preds,class_preds,conf_preds

class CustomizableVFL:
    def __init__(
        self,
        num_clients: int,
        feature_splits: List[List[int]],
        data_alignment: DataAlignment,
        client_models_config: List[Dict],
        top_model_config: Dict,
        embedding_size: int = 8,
        mixup_strategy: MixupStrategy = MixupStrategy.NO_MIXUP,
        device: str = 'cuda'
    ):
        self.num_clients = num_clients
        self.feature_splits = feature_splits
        self.data_alignment = data_alignment
        self.device = device
        self.embedding_size = embedding_size
        self.mixup_strategy = mixup_strategy
        
        # Initialize client models
        self.client_models = []
        self.client_optimizers = []
        
        for i in range(num_clients):
            config = client_models_config[i]
            if config.get('model_type') == 'resnet50':
                model = ResNet50ClientModel(
                    input_size=len(feature_splits[0]),
                    embedding_size=embedding_size,
                    image_width=config.get('image_width', None),
                    pretrained=config.get('pretrained', True)
                ).to(device)
            else:
                model = ClientModel(
                    input_size=len(feature_splits[i]),
                    hidden_layers=config['hidden_layers'],
                    embedding_size=embedding_size
                ).to(device)
            self.client_models.append(model)
            self.client_optimizers.append(
                optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.0001))
            )
        
        # Initialize top model

        # if config.get('server')=='object':
        self.top_model = ResNetBBoxPredictor(
             num_clients=num_clients,
             total_embedding_size=num_clients*embedding_size,
             num_classes=80,
             num_boxes=20,
             conf_threshold=0.7
         ).to(device)
        # else:
        #     self.top_model = ServerModel(
        #     num_clients=num_clients,
        #     embedding_size=embedding_size,
        #     hidden_layers=top_model_config['hidden_layers'],
        #     output_size=top_model_config.get('output_size', 1)
        #     ).to(device)

        self.top_optimizer = optim.Adam(
            self.top_model.parameters(),
            lr=top_model_config.get('learning_rate', 0.0001)
        )

        # Mixup strategy for labels
        if self.mixup_strategy == MixupStrategy.NO_MIXUP:
            self.mixup_fn = self.no_mixup
        elif self.mixup_strategy == MixupStrategy.MAX_MIXUP:
            self.mixup_fn = self.max_mixup
        elif self.mixup_strategy == MixupStrategy.MEAN_MIXUP:
            self.mixup_fn = self.mean_mixup
        elif self.mixup_strategy == MixupStrategy.IMPORTANCE_MIXUP:
            self.mixup_fn = self.importance_mixup
        elif self.mixup_strategy == MixupStrategy.MODEL_BASED_MIXUP:
            self.mixup_fn = self.model_based_mixup
        elif self.mixup_strategy == MixupStrategy.MUTUAL_INFO_MIXUP:
            self.mixup_fn = self.mutual_info_mixup
        
        self.loss_fn = object_detection_loss

    @staticmethod
    def no_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor] = None
    ) -> torch.Tensor:
        return client_batch_labels[0]
    
    @staticmethod
    def max_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.max(torch.stack(client_batch_labels), dim=0).values
    
    @staticmethod
    def mean_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.mean(torch.stack(client_batch_labels), dim=0)

    @staticmethod
    def importance_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        
        # set_trace()
        embeddings = [embedding.detach().cpu().numpy() for embedding in client_embeddings]
        client_importance = {}
    
        for client_id, embedding in enumerate(embeddings):
            correlations = []
            for i in range(embedding.shape[1]):  # Iterate over embedding dimensions
                corr = np.corrcoef(embedding[:, i], client_batch_labels[client_id].cpu().numpy().squeeze())[0, 1]
                correlations.append(abs(corr))  # Use absolute correlation
            
            # Aggregate correlations for this client
            client_importance[client_id] = np.mean(correlations)
    
        # Normalize importance scores to sum to 1
        total_importance = sum(client_importance.values())
        for client_id in client_importance:
            client_importance[client_id] /= total_importance

        # set_trace()
        # Weighted average of labels
        weighted_labels = torch.zeros_like(client_batch_labels[0])
        for client_id, importance in client_importance.items():
            weighted_labels += importance * client_batch_labels[client_id]
        
        return weighted_labels
    
    @staticmethod
    def model_based_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor],
        model_type: str = 'linear'
    ) -> torch.Tensor:
        
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        client_importance = {}
        
        # Calculate importance for each client using their own labels
        for client_id, (embedding, labels) in enumerate(zip(client_embeddings, client_batch_labels)):
            # Convert to numpy arrays
            embeddings_np = embedding.detach().cpu().numpy()
            labels_np = labels.cpu().numpy().squeeze()
            
            if model_type == 'linear':
                model = LinearRegression()
                model.fit(embeddings_np, labels_np)
                importances = np.abs(model.coef_)  # Use absolute coefficients
            elif model_type == 'random_forest':
                model = RandomForestRegressor()
                model.fit(embeddings_np, labels_np)
                importances = model.feature_importances_
            
            # Aggregate feature importances for this client
            client_importance[client_id] = np.mean(importances)
        
        # Normalize importance scores to sum to 1
        total_importance = sum(client_importance.values())
        for client_id in client_importance:
            client_importance[client_id] /= total_importance
        
        # Create weighted average of labels using calculated importances
        weighted_labels = torch.zeros_like(client_batch_labels[0])
        for client_id, importance in client_importance.items():
            weighted_labels += importance * client_batch_labels[client_id]
        
        return weighted_labels
    
    @staticmethod
    def mutual_info_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        
        from sklearn.feature_selection import mutual_info_regression
        
        client_importance = {}
        
        # Calculate importance for each client using mutual information
        for client_id, (embedding, labels) in enumerate(zip(client_embeddings, client_batch_labels)):
            # Convert to numpy arrays
            embeddings_np = embedding.detach().cpu().numpy()
            labels_np = labels.cpu().numpy().squeeze()
            
            # Calculate mutual information between embeddings and labels
            mi_scores = mutual_info_regression(embeddings_np, labels_np)
            
            # Aggregate mutual information scores for this client
            client_importance[client_id] = np.mean(mi_scores)
        
        # Normalize importance scores to sum to 1
        total_importance = sum(client_importance.values())
        for client_id in client_importance:
            client_importance[client_id] /= total_importance
        
        # Create weighted average of labels using calculated importances
        weighted_labels = torch.zeros_like(client_batch_labels[0])
        for client_id, importance in client_importance.items():
            weighted_labels += importance * client_batch_labels[client_id]
        
        return weighted_labels




    def prepare_datasets(
        self,
        X: np.ndarray,
        y: List[List[int]],
        subset_size: Optional[int] = None,
        train_size: float = 0.7,
        unaligned_ratio: float = 0.8,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
        """
        Prepare datasets for aligned or unaligned cases with variable-length labels.
        Returns tuples of training/testing features and lists of label tensors.
        """
        client_data = []
        client_labels = []

        if subset_size:
            X = X[:subset_size]
            y = y[:subset_size]
        
        # Convert y to a list of tensors for variable-length labels
        y_tensors = [torch.tensor(labels, dtype=torch.float32).to(self.device) for labels in y]

        if self.data_alignment == DataAlignment.ALIGNED:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y_tensors, train_size=train_size, shuffle=True
            )
            
            for feature_split in self.feature_splits:
                (s,e)=feature_split[1]
                print(X_train_full[0].shape)
                X_train = torch.stack([
                torch.tensor(img[:, s:e :], dtype=torch.float32).to(self.device) for img in X_train_full
                ])
                X_test = torch.stack([
                torch.tensor(img[:, s:e, :], dtype=torch.float32).to(self.device) for img in X_test_full
                ])
                y_train_list = [y_train[i] for i in range(len(y_train))]
                y_test_list = [y_test[i] for i in range(len(y_test))]

                client_data.append((X_train, X_test))
                client_labels.append((y_train_list, y_test_list))
            
        elif self.data_alignment == DataAlignment.UNALIGNED:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y_tensors, train_size=train_size, shuffle=True
            )

            all_indices_random = np.random.permutation(len(X_train_full))
            unaligned_indices = all_indices_random[:int(len(X_train_full) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X_train_full)), unaligned_indices)
            
            for feature_split in self.feature_splits:
                train_unaligned_indices = np.random.permutation(unaligned_indices)
                train_aligned_indices = aligned_indices
                client_indices = np.concatenate([train_aligned_indices, train_unaligned_indices])
                
                X_train = torch.tensor(X_train_full[:, feature_split], dtype=torch.float32).to(self.device)
                X_train = X_train[list(client_indices)]
                X_test = torch.tensor(X_test_full[:, feature_split], dtype=torch.float32).to(self.device)
                
                y_train_list = [y_train[i] for i in client_indices]
                y_test_list = [y_test[i] for i in range(len(y_test))]
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_list, y_test_list))
        
        else:
            all_indices_random = np.random.permutation(len(X))
            unaligned_indices = all_indices_random[:int(len(X) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X)), unaligned_indices)
            
            for feature_split in self.feature_splits:
                X_train, X_test, y_train, y_test = train_test_split(
                    X[:, feature_split],
                    y_tensors,
                    train_size=train_size,
                    shuffle=False
                )
                
                X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                
                y_train_list = [y_train[i] for i in range(len(y_train))]
                y_test_list = [y_test[i] for i in range(len(y_test))]
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_list, y_test_list))

    
        
        return client_data, client_labels


    def train_step(self, client_batch_data: List[torch.Tensor], client_batch_labels: List[torch.Tensor]) -> float:
        """Modified training step with top model coordination"""

        for optimizer in self.client_optimizers:
            optimizer.zero_grad()
        self.top_optimizer.zero_grad()

        client_embeddings = []
        for model, data in zip(self.client_models, client_batch_data):
            embedding = model(data)  # This gives embeddings of size [2, 64] for each client
            client_embeddings.append(embedding)

        # Concatenate all client embeddings along dimension 1
        concatenated_embeddings = torch.cat(client_embeddings, dim=1)  # This will give [2, 128] for 2 clients with 64-dim embeddings

        # Pass concatenated embeddings to top model
        box,labels,conf = self.top_model(concatenated_embeddings)
        truth=self.no_mixup(client_batch_labels,concatenated_embeddings)
        
        label_real = [tensor[:, 0] for tensor in truth]  # Extract first column as labels
        box_real = [tensor[:, 1:] for tensor in truth]  # Extract remaining columns as boxes


        loss = self.loss_fn(box,labels,conf,box_real,label_real,iou_threshold=0.5)
        loss.backward()
        
    
        for optimizer in self.client_optimizers:
            optimizer.step()
        self.top_optimizer.step()
        
        return loss.item()

    def evaluate(self, client_test_data: List[torch.Tensor], y_test: torch.Tensor) -> float:
        """Evaluate using top model"""
        with torch.no_grad():
            # Get embeddings from client models
            client_embeddings = [model(data) for model, data in zip(self.client_models, client_test_data)]
            
            # Concatenate embeddings along dimension 1
            concatenated_embeddings = torch.cat(client_embeddings, dim=1)  # Shape: [batch_size, total_embedding_dim]
            
            # Pass concatenated embeddings to top model
            box, labels, conf = self.top_model(concatenated_embeddings)
            
            # Get ground truth
            truth = self.no_mixup(y_test, concatenated_embeddings)
            
            truth=y_test
            
        
            # Extract first column as labels and remaining columns as boxes
            label_real = [tensor[:, 0] for tensor in truth]  
            box_real = [tensor[:, 1:] for tensor in truth]

            # Compute loss
            loss = self.loss_fn(box, labels, conf, box_real, label_real, iou_threshold=0.5)
            
            return float(loss)



    def train(
        self,
        client_data: List[Tuple[torch.Tensor, torch.Tensor]],
        client_labels: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int,
        batch_size: int
    ) -> Dict:
        best_mse = np.inf
        best_weights = None
        history = []
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            min_train_length = min(len(data[0]) for data in client_data)
            batch_start = torch.arange(0, min_train_length, batch_size) # choosing the starting index of each batch
            
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    client_batch_data = [
                        train_data[start:start+batch_size]
                        for train_data, _ in client_data
                    ]
                    client_batch_labels = [
                        train_labels[start:start+batch_size]
                        for train_labels, _ in client_labels
                    ]

                    loss = self.train_step(client_batch_data, client_batch_labels)
                    total_loss += loss
                    bar.set_postfix(mse=float(loss))
            
            print(f"Epoch {epoch}, Average MSE: {total_loss/len(batch_start)}")
            # set_trace()
            min_test_length = min(len(test_data) for _, test_data in client_data)
            test_data = [test_data[:min_test_length] for _, test_data in client_data]
            mse = self.evaluate(test_data, client_labels[0][1][:min_test_length])
            print(f"Epoch {epoch}, Test RMSE: {math.sqrt(mse)}")
            history.append(mse)
            
            if mse < best_mse:
                best_mse = mse
                best_weights = {
                    'client_models': [copy.deepcopy(model.state_dict()) for model in self.client_models],
                    'top_model': copy.deepcopy(self.top_model.state_dict())
                }
        
        if best_weights:
            for model, weights in zip(self.client_models, best_weights['client_models']):
                model.load_state_dict(weights)
            self.top_model.load_state_dict(best_weights['top_model'])
        
        return {
            'best_mse': best_mse,
            'history': history
        }

from typing import List, Optional, Tuple

def split_features(num_features: int, num_clients: int, distribution: Optional[List[int]] = None) -> List[Tuple[int, int]]:
    if distribution is None:
        # If no distribution is provided, divide features evenly
        num_clients = 2  # Default number of clients
        base_size = num_features // num_clients
        remainder = num_features % num_clients
        
        distribution = [base_size] * num_clients
        for i in range(remainder):
            distribution[i] += 1
    
    if sum(distribution) != num_features:
        raise ValueError("Distribution does not match the total number of features.")
    
    feature_splits = []
    start = 0
    for size in distribution:
        feature_splits.append((start, start + size))  # Store range as (start, end)
        start += size
    
    return feature_splits




def fetch_data():
    df = pd.read_csv('Datasets/MiningProcess_Flotation_Plant_Database.csv', skiprows=1)
  #  set_trace()
    df = df.drop(df.columns[0], axis=1)  # Drop the first column

    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)

    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]

    data = data.to_numpy()
    target = target.to_numpy()
    # set_trace()
    return data, target, data.shape[1]

def modify_bbox_coordinates(bndbox, original_image_size, new_image_size):

    bndbox['xmin'] = int(int(bndbox['xmin']) * new_image_size[1] / original_image_size[1])
    bndbox['xmax'] = int(int(bndbox['xmax']) * new_image_size[1] / original_image_size[1])
    bndbox['ymin'] = int(int(bndbox['ymin']) * new_image_size[0] / original_image_size[0])
    bndbox['ymax'] = int(int(bndbox['ymax']) * new_image_size[0] / original_image_size[0])

    return bndbox




def fetch_ice_pets():
    """
         Take the image split it into two parts vertically and bounding boxes and put them in the list
         Return: Images and labels in a splittled format aligend with each other at indexes
    """
    
    images_path = 'Datasets/ice_pets/images/'
    labels_path = 'Datasets/ice_pets/annotations/xmls/'

    images = []
    labels = []
    image_size = (400, 600)


    #sorted list of images
    for filename in sorted(os.listdir(labels_path)):
        # print(filename)

        image_name = filename.split('.')[0] + '.jpg'
        # open image and save it to a list
        image_np = cv2.imread(images_path + image_name)
        # if corrupt image
        if image_np is None:
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        original_image_size = image_np.shape
        image_np = cv2.resize(image_np, image_size)
        image_np = image_np.flatten()
        # add to list
        images.append(image_np)

        with open(labels_path + filename) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

            if isinstance(data_dict['annotation']['object'], list):
                bndbox = data_dict['annotation']['object'][0]['bndbox']
            else:
                bndbox = data_dict['annotation']['object']['bndbox']

            bndbox = modify_bbox_coordinates(bndbox, original_image_size, image_size)
            label = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
            labels.append(label)

    images = np.vstack(images)
    labels = np.array(labels)
    
    return images, labels, image_size

def run_program():

    # Parse the Arguments
    parser = argparse.ArgumentParser(description='Customize how you want to run VFL')
    parser.add_argument('--unaligned', action='store_true', help='Whether to run VFL with aligned data')
    parser.add_argument('--unaligned_ratio', type=float, default=0.8, help='Ratio of unaligned data')
    parser.add_argument('--mixup_strategy', type=str, default='no_mixup', help='Mixup strategy to use', choices=['no_mixup', 'max_mixup', 'mean_mixup', 'importance_mixup', 'model_based_mixup', 'mutual_info_mixup'])
    
    args = parser.parse_args()

    algn_type = DataAlignment.ALIGNED if not args.unaligned else DataAlignment.UNALIGNED
    mixup_strategy = MixupStrategy(args.mixup_strategy)
    unaligned_ratio = args.unaligned_ratio
    print(f"Mixup Strategy: {mixup_strategy}")
    print(f"Data Alignment: {algn_type}")
    
    # Load data
    X,y,feat_no= load_coco8_data()
    print("data loaded")
    
    # Configuration
    num_clients = 2
    (h,w,c)=X[0].shape
    feat_no = w
    a = split_features(feat_no, num_clients)
    feature_splits=[]
    for i in a:
        print(i)
        feature_splits.append((h,i,c))

    # feature_splits = [
    #     [0, 1, 2, 3, 4, 5, 6, 7],  # Client 1 features
    #     [8, 9, 10, 11, 12, 13, 14],  # Client 2 features
    #     [15, 16, 17, 18, 19, 20, 21]      # Client 3 features
    # ]
    
    # Client model configurations
    client_models_config = [
        {'hidden_layers': [12, 6], 'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600},
        {'hidden_layers': [12, 6], 'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600},
        {'hidden_layers': [8, 4], 'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600}
    ]

    print("client config loaded")
      
    # Top model configuration
    top_model_config = {
        'hidden_layers': [24, 12],
        'learning_rate': 0.001,
        'server':'object'
    }


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize VFL system
    vfl = CustomizableVFL(
        num_clients=num_clients,
        feature_splits=feature_splits,
        data_alignment=algn_type,
        client_models_config=client_models_config,
        top_model_config=top_model_config,
        embedding_size=64,
        mixup_strategy=mixup_strategy,
        device=device
    )

    print("VFL initialized")
    

    client_data, client_labels = vfl.prepare_datasets(X, y, subset_size=250, train_size=0.8, unaligned_ratio=unaligned_ratio)

    print("data prepared")
    
    # Train the system
    results = vfl.train(
        client_data=client_data,
        client_labels=client_labels,
        n_epochs=100,
        batch_size=100
    )
    
    print(f"Final Best MSE: {results['best_mse']}")


seed = 42  # Choose any fixed number
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    run_program()