import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from enum import Enum
from typing import List, Tuple, Dict, Optional, Callable
import copy
import tqdm
import argparse
#from sklearn.preprocessing import LabelEncoder
import random
import pandas as pd
import math
from typing import List, Optional
from torchvision import models
import cv2
import xmltodict
import os
import wandb
import yaml
import sys
import time

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
        
        if image_width is None:
            self.image_width = int(math.sqrt(input_size))
        else:
            self.image_width = image_width
            
        # Calculate height based on input size and width
        self.image_height = math.ceil(input_size / self.image_width)
        
        # Load a pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the first layer to accept single-channel input instead of 3-channel (RGB)
        if pretrained:
            first_conv_weights = self.resnet.conv1.weight.data.clone()
            # Average the weights across the RGB channels to create weights for a single channel
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.resnet.conv1.weight.data = torch.mean(first_conv_weights, dim=1, keepdim=True)
        else:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the classification layer
        
        self.fc_embedding = nn.Linear(num_features, embedding_size)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Pad the input to fit the image dimensions if needed
        padded_size = self.image_width * self.image_height
        if self.input_size < padded_size:
            padding = torch.zeros(batch_size, padded_size - self.input_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        x = x.view(batch_size, 1, self.image_height, self.image_width)
        
        # Ensure the input is large enough for ResNet (which typically expects at least 224x224)
        # If the image is too small, we'll upsample it
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
    def __init__(self, num_clients: int, embedding_size: int, hidden_layers: List[int], aggregate_fn: Callable = None):
        super().__init__()
        
        self.aggregate_fn = aggregate_fn

        if self.aggregate_fn is not  None:
            total_input_size = embedding_size
        else:
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
        if self.aggregate_fn is not None:
            combined = self.aggregate_fn(torch.stack(embeddings), dim=0)
        else:
            combined = torch.cat(embeddings, dim=1)

        return self.layers(combined)

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
        self.client_schedulers = []
        
        for i in range(num_clients):
            config = client_models_config[i]
            if config.get('model_type', 'mlp') == 'resnet50':
                model = ResNet50ClientModel(
                    input_size=len(feature_splits[i]),
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
            self.client_schedulers.append(
                optim.lr_scheduler.CosineAnnealingLR(
                    self.client_optimizers[i],
                    T_max=config.get('n_epochs', 100)
                )
            )
        
        # Initialize top model with aggregate function if specified
        aggregate_fn = None
        if top_model_config.get('aggregate_fn') == 'mean':
            aggregate_fn = torch.mean
            
        self.top_model = ServerModel(
            num_clients=num_clients,
            embedding_size=embedding_size,
            hidden_layers=top_model_config['hidden_layers'],
            aggregate_fn=aggregate_fn
        ).to(device)

        self.top_optimizer = optim.Adam(
            self.top_model.parameters(),
            lr=top_model_config.get('learning_rate', 0.0001)
        )

        self.top_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.top_optimizer,
            T_max=top_model_config.get('n_epochs', 100)
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
        
        self.loss_fn = nn.MSELoss()

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
        y: np.ndarray,
        subset_size: Optional[int] = None,
        train_size: float = 0.7,
        unaligned_ratio: float = 0.8
        ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:

        """Prepare datasets with normalization for client models"""
        client_data = []
        client_labels = []

        if subset_size:
            X = X[:subset_size]
            y = y[:subset_size]

        if self.data_alignment == DataAlignment.ALIGNED:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=True
            )
            
            for feature_split in self.feature_splits:
                # Extract features for this client
                X_train_client = X_train_full[:, feature_split]
                X_test_client = X_test_full[:, feature_split]
                
                # Normalize the data for this client
                # Calculate mean and std on training data
                mean = np.mean(X_train_client, axis=0)
                std = np.std(X_train_client, axis=0)
                # Replace zero std with 1 to avoid division by zero
                std = np.where(std == 0, 1.0, std)
                
                # Apply normalization
                X_train_normalized = (X_train_client - mean) / std
                X_test_normalized = (X_test_client - mean) / std
                
                # Convert to tensors
                X_train = torch.tensor(X_train_normalized, dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test_normalized, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_tensor, y_test_tensor))

        elif self.data_alignment == DataAlignment.UNALIGNED:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=True
            )

            all_indices_random = np.random.permutation(len(X_train_full))
            unaligned_indices = all_indices_random[:int(len(X_train_full) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X_train_full)), unaligned_indices)
            
            for feature_split in self.feature_splits:
                train_unaligned_indices = unaligned_indices.copy()
                np.random.shuffle(unaligned_indices)
                train_aligned_indices = aligned_indices
                client_indices = np.concatenate([train_aligned_indices, train_unaligned_indices])

                # Extract features for this client
                X_train_client = X_train_full[:, feature_split]
                X_test_client = X_test_full[:, feature_split]
                
                # Normalize the data
                mean = np.mean(X_train_client, axis=0)
                std = np.std(X_train_client, axis=0)
                # Replace zero std with 1 to avoid division by zero
                std = np.where(std == 0, 1.0, std)
                
                X_train_normalized = (X_train_client - mean) / std
                X_test_normalized = (X_test_client - mean) / std
                
                # Apply client indices after normalization
                X_train_normalized = X_train_normalized[list(client_indices)]
                y_train_client = y_train[list(client_indices)]
                
                # Convert to tensors
                X_train = torch.tensor(X_train_normalized, dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test_normalized, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train_client, dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_tensor, y_test_tensor))

        else:
            all_indices_random = np.random.permutation(len(X))
            unaligned_indices = all_indices_random[:int(len(X) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X)), unaligned_indices)

            for feature_split in self.feature_splits:
                X_client = X[:, feature_split]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_client,
                    y,
                    train_size=train_size,
                    shuffle=False
                )
                
                # Normalize the data
                mean = np.mean(X_train, axis=0)
                std = np.std(X_train, axis=0)
                # Replace zero std with 1 to avoid division by zero
                std = np.where(std == 0, 1.0, std)
                
                X_train_normalized = (X_train - mean) / std
                X_test_normalized = (X_test - mean) / std
                
                # Convert to tensors
                X_train = torch.tensor(X_train_normalized, dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test_normalized, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_tensor, y_test_tensor))
        
        return client_data, client_labels

    def train_step(self, client_batch_data: List[torch.Tensor], client_batch_labels: List[torch.Tensor]) -> float:
        """Modified training step with top model coordination"""
        # Reset gradients

        for optimizer in self.client_optimizers:
            optimizer.zero_grad()
        self.top_optimizer.zero_grad()
        
        # Forward pass through client models to get embeddings
        client_embeddings = []
        for model, data in zip(self.client_models, client_batch_data):
            embedding = model(data)
            client_embeddings.append(embedding)
        
        # Forward pass through top model
        final_prediction = self.top_model(client_embeddings)
        
        # Calculate loss (using labels from first client if unaligned)
        loss = self.loss_fn(final_prediction, self.mixup_fn(client_batch_labels, client_embeddings))
        
        # Backward pass
        loss.backward()
        
        # Update weights
        for optimizer in self.client_optimizers:
            optimizer.step()
        self.top_optimizer.step()
        
        return loss.item()

    def evaluate(self, client_test_data: List[torch.Tensor], y_test: torch.Tensor) -> float:
        """Evaluate using top model"""
        with torch.no_grad():
            # Get embeddings from client models
            client_embeddings = []
            for model, data in zip(self.client_models, client_test_data):
                embedding = model(data)
                client_embeddings.append(embedding)
            
            # Get prediction from top model
            final_prediction = self.top_model(client_embeddings)
            
            mse = self.loss_fn(final_prediction, y_test)
            return float(mse)

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

        # save running time to wandb
        running_time = 0

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
                    
                    start_time = time.time()
                    loss = self.train_step(client_batch_data, client_batch_labels)
                    end_time = time.time()
                    running_time += end_time - start_time

                    total_loss += loss
                    bar.set_postfix(mse=float(loss))

            # Step the learning rate scheduler
            for scheduler in self.client_schedulers:
                scheduler.step()
                
            self.top_scheduler.step()
            
            # Evaluate on test data
            min_test_length = min(len(test_data) for _, test_data in client_data)
            test_data = [test_data[:min_test_length] for _, test_data in client_data]
            mse = self.evaluate(test_data, client_labels[0][1][:min_test_length])
            print(f"Epoch {epoch}, Test RMSE: {math.sqrt(mse)}")
            
            # Log to wandb if enabled
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": total_loss/len(batch_start),
                    "test_mse": mse,
                    "test_rmse": math.sqrt(mse)
                })
            
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

        if wandb.run is not None:
            wandb.log({
                "running_time": running_time,
                "best_rmse": math.sqrt(best_mse)
            })
            
        return {
            'best_mse': best_mse,
            'history': history
        }

def split_features(num_features: int, num_clients: int, distribution: Optional[List[int]] = None) -> List[List[int]]:
    if distribution is None:
        # If no distribution is provided, divide features evenly
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
        feature_splits.append(list(range(start, start + size)))
        start += size
    
    return feature_splits

# Dataset loading functions

def real_estate():
    df=pd.read_csv('Datasets/REAL_ESTATE.csv',skiprows=1)
    df = df.drop(df.columns[:2], axis=1)
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    target = df[df.columns[-1]]
    data = df[df.columns[:-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def concrete():
    df = pd.read_csv("Datasets/CONCRETE_COMPRESSIVE_STRENGTH.csv")
    target = df[df.columns[-1]]
    data = df[df.columns[:-1]]
    print(data.isnull().sum())
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def energy():
    df = pd.read_csv("Datasets/ENERGY_EFFICIENCY.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    df = df.drop(df.columns[-1],axis=1)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]


def yacht_hydrodynamics():
    df = pd.read_csv("Datasets/YACHT.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]

def superconductivity():
    df = pd.read_csv("Datasets/SUPERCONDUCTIVITY.csv")
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)
    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]
    data = data.to_numpy()
    target = target.to_numpy()
    return data, target, data.shape[1]
