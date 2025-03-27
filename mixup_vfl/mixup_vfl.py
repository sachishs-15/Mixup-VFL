from typing import List, Tuple, Dict, Optional
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.mixup_strategies import no_mixup, max_mixup, mean_mixup, importance_mixup, model_based_mixup, mutual_info_mixup,add_mixup
from config.config import DataAlignment, MixupStrategy
from models import *
import time
import tqdm
import copy
import wandb
import math
import os
from sklearn.model_selection import train_test_split

class MixupVFL_Regression:

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
        self.top_model=None
        
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
                model = MLPClientModel(
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
        
            
        self.top_model = ServerModel_Regression(
            num_clients=num_clients,
            embedding_size=embedding_size,
            hidden_layers=top_model_config['hidden_layers'],
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
            self.mixup_fn = no_mixup
        elif self.mixup_strategy == MixupStrategy.MAX_MIXUP:
            self.mixup_fn = max_mixup
        elif self.mixup_strategy == MixupStrategy.MEAN_MIXUP:
            self.mixup_fn = mean_mixup
        elif self.mixup_strategy == MixupStrategy.IMPORTANCE_MIXUP:
            self.mixup_fn = importance_mixup
        elif self.mixup_strategy == MixupStrategy.MODEL_BASED_MIXUP:
            self.mixup_fn = model_based_mixup
        elif self.mixup_strategy == MixupStrategy.MUTUAL_INFO_MIXUP:
            self.mixup_fn = mutual_info_mixup
        
        self.loss_fn = nn.MSELoss()


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
    
from utils.loss import object_detection_loss
from models import ResNet50ClientModel, ResNeXt29ClientModel, MLPClientModel, ServerModel_Regression, ResNetBBoxPredictor

class MixupVFL_ObjectDetection:

    def __init__(
        self,
        num_clients: int,
        feature_splits: List[List[int]],
        data_alignment: DataAlignment,
        client_models_config: List[Dict],
        top_model_config: Dict,
        embedding_size: int = 8,
        mixup_strategy: MixupStrategy = MixupStrategy.NO_MIXUP,
        device: str = 'cuda',
    ):
        self.num_clients = num_clients
        self.feature_splits = feature_splits
        self.data_alignment = data_alignment
        self.device = device
        self.embedding_size = embedding_size
        self.mixup_strategy = mixup_strategy
        self.X_val=None
        self.Y_val=None

        # Initialize client models
        self.client_models = []
        self.client_optimizers = []

        for i in range(num_clients):
            config = client_models_config[i]
            if config.get('model_type') == 'resnet50':
                model = ResNet50ClientModel(
                    embedding_size=embedding_size,
                    pretrained=config.get('pretrained', True)
                ).to(device)
            elif config.get('model_type') == 'resnext50':
                model = ResNeXt29ClientModel(
                    embedding_size=embedding_size,
                    pretrained=config.get('pretrained', True)
                ).to(device)
            else:
                model = MLPClientModel(
                    input_size=len(feature_splits[i]),
                    hidden_layers=config['hidden_layers'],
                    embedding_size=embedding_size
                ).to(device)
            self.client_models.append(model)
            self.client_optimizers.append(
                optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.0001))
            )
        cmm=self.client_models

        # Initialize top model

        self.top_model = ResNetBBoxPredictor(
                num_clients=num_clients,
                total_embedding_size=num_clients*embedding_size,
                num_classes=91,
                num_boxes=20,
                conf_threshold=0.7
            ).to(device)
        tp=self.top_model

        self.top_optimizer = optim.Adam(
            self.top_model.parameters(),
            lr=top_model_config.get('learning_rate', 0.0001)
        )

        # Mixup strategy for labels


        self.loss_fn = object_detection_loss

    def reverse_image_transform(self,transformed_tensor):
        """
        Reverse the normalization and convert tensor back to numpy array for visualization

        Args:
            transformed_tensor: Normalized tensor from the dataloader [C, H, W]

        Returns:
            numpy_image: Denormalized image as numpy array [H, W, C] with values in [0, 255]
        """
        # Reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        denormalized = transformed_tensor * std + mean

        # Clamp values to [0, 1]
        denormalized = torch.clamp(denormalized, 0, 1)

        # Convert to numpy and change dimension order
        numpy_image = denormalized.permute(1, 2, 0).cpu().numpy()

        # Convert to uint8 range [0, 255]
        numpy_image = (numpy_image * 255).astype(np.uint8)

        return numpy_image

    def convert_normalized_bbox_to_wandb_format(self,bbox, image_width, image_height):
        """
        Convert normalized bbox coordinates [x_center, y_center, width, height] to
        wandb format with absolute coordinates [minX, minY, maxX, maxY]

        Args:
            bbox: Tensor or list with [x_center, y_center, width, height] in normalized form [0,1]
            image_width: Width of the image
            image_height: Height of the image

        Returns:
            Dictionary with wandb box format
        """
        # Convert from normalized [0,1] to absolute coordinates
        x_center, y_center, width, height = bbox

        # Convert to absolute coordinates
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        width_abs = width * image_width
        height_abs = height * image_height

        # Calculate corners
        min_x = x_center_abs - (width_abs / 2)
        min_y = y_center_abs - (height_abs / 2)
        max_x = x_center_abs + (width_abs / 2)
        max_y = y_center_abs + (height_abs / 2)

        return {
            "position": {
                "minX": float(min_x),
                "minY": float(min_y),
                "maxX": float(max_x),
                "maxY": float(max_y)
            }
        }


    def prepare_datasets(
        self,
        X: np.ndarray,
        y: List[List[int]],
        X_test_full: np.ndarray,
        y_test_full: List[List[int]],
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

        # Convert y to a list of tensors for variable-length labels
        y_tensors = [torch.tensor(labels, dtype=torch.float32).to(self.device) for labels in y]

        if self.data_alignment == DataAlignment.ALIGNED:
            X_train_full = X
            y_train_full = y
            X_test = X_test_full
            y_test = y_test_full

            for feature_split in self.feature_splits:
                (s, e) = feature_split[2]
                X_train = torch.stack([
                    torch.tensor(img[:, :, s:e], dtype=torch.float32).to(self.device) for img in X_train_full
                ])
                X_test = torch.stack([
                    torch.tensor(img[:, :, s:e], dtype=torch.float32).to(self.device) for img in X_test_full
                ])

                y_train_list = [y_train_full[i] for i in range(len(y_train_full))]
                y_test_list = [y_test[i] for i in range(len(y_test))]

                client_data.append((X_train, X_test))
                client_labels.append((y_train_list, y_test_list))

        elif self.data_alignment == DataAlignment.UNALIGNED:
            X_train_full = X
            y_train_full = y
            X_test = X_test_full
            y_test = y_test_full

            # Generate indices for unaligned data
            all_indices_random = np.random.permutation(len(X_train_full))
            unaligned_indices = all_indices_random[:int(len(X_train_full) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X_train_full)), unaligned_indices)

            for feature_split in self.feature_splits:
                (s, e) = feature_split[2]  # Fixed: extract the start and end indices

                # Shuffle unaligned indices for this client
                train_unaligned_indices = np.random.permutation(unaligned_indices)
                train_aligned_indices = aligned_indices
                client_indices = np.concatenate([train_aligned_indices, train_unaligned_indices])

                # Apply the same dimension splitting as in the aligned case
                X_train = torch.stack([
                    torch.tensor(img[:, :, s:e], dtype=torch.float32).to(self.device) for img in X_train_full
                ])
                X_train = X_train[list(client_indices)]

                X_test = torch.stack([
                    torch.tensor(img[:, :, s:e], dtype=torch.float32).to(self.device) for img in X_test_full
                ])

                y_train_list = [y_train_full[i] for i in client_indices]
                y_test_list = [y_test[i] for i in range(len(y_test))]

                client_data.append((X_train, X_test))
                client_labels.append((y_train_list, y_test_list))

        return client_data, client_labels
    
    def train_step(self, client_batch_data: List[torch.Tensor], client_batch_labels: List[torch.Tensor]) -> float:
        """Modified training step with top model coordination and gradient monitoring"""

        for optimizer in self.client_optimizers:
            optimizer.zero_grad()
        self.top_optimizer.zero_grad()

        client_embeddings = []
        for model, data in zip(self.client_models, client_batch_data):
            embedding = model(data)
            client_embeddings.append(embedding)

        concatenated_embeddings = torch.cat(client_embeddings, dim=1)

        box, labels, conf = self.top_model(concatenated_embeddings)
        truth = add_mixup(client_batch_labels, concatenated_embeddings)

        label_real = [tensor[:, 0] for tensor in truth]
        box_real = [tensor[:, 1:] for tensor in truth]

        loss, cl, bl, cfl = self.loss_fn(box, labels, conf, box_real, label_real)

        # Compute gradients
        loss.backward()

        # Check for vanishing gradients
        client_grad_norms = []
        for i, model in enumerate(self.client_models):
            model_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    model_grad_norm += param_norm
            client_grad_norms.append(model_grad_norm)
            wandb.log({f"client_{i}_total_grad_norm": model_grad_norm})

        # Check top model gradients
        top_model_grad_norm = 0
        for name, param in self.top_model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                top_model_grad_norm += param_norm

        wandb.log({"top_model_total_grad_norm": top_model_grad_norm})

        # Perform optimization step
        for optimizer in self.client_optimizers:
            optimizer.step()
        self.top_optimizer.step()

        return loss.item(), cl.item(), bl.item(), cfl.item()
    
    def evaluate(self, client_test_data: List[torch.Tensor], y_test: torch.Tensor, save=False) -> float:
        """Evaluate using top model and log images with bounding boxes to wandb"""
        with torch.no_grad():
            client_embeddings = [model(data) for model, data in zip(self.client_models, client_test_data)]
            concatenated_embeddings = torch.cat(client_embeddings, dim=1)
            box, labels, conf = self.top_model(concatenated_embeddings)
            truth = y_test
            label_real = [tensor[:, 0] for tensor in truth]
            box_real = [tensor[:, 1:] for tensor in truth]
            loss, _, _, _ = self.loss_fn(box, labels, conf, box_real, label_real)

            # Log images with separate ground truth and prediction boxes to wandb
            if save:
                for i in range(min(10, len(self.X_val))):
                    # Original image without normalization
                    img_tensor = self.X_val[i]

                    # Reverse the normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)

                    # Convert to numpy for wandb
                    img = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)

                    # Get image dimensions
                    img_height, img_width = img.shape[0], img.shape[1]

                    # PREDICTION BOXES
                    pred_boxes_data = []

                    pred_boxes_data1=[]
                    if len(box) > i and box[i] is not None:
                        for j in range(len(box[i])):
                            # Show all predictions regardless of confidence for debugging
                            coords = box[i][j].tolist()

                            # Assuming [x_center,y_center,width,height] normalized format
                            x_center, y_center, width, height = coords

                            # Convert from normalized [0,1] to absolute pixels
                            x_center_abs = x_center * img_width
                            y_center_abs = y_center * img_height
                            width_abs = width * img_width
                            height_abs = height * img_height

                            # Calculate corners for wandb format
                            min_x = max(0, x_center_abs - (width_abs / 2))
                            min_y = max(0, y_center_abs - (height_abs / 2))
                            max_x = min(img_width, x_center_abs + (width_abs / 2))
                            max_y = min(img_height, y_center_abs + (height_abs / 2))

                            # Check for valid box
                            if max_x > min_x and max_y > min_y and conf[i][j].item()>0.3:
                                confidence = conf[i][j].item()
                                class_id = int(labels[i][j].argmax().item())
                                pred_boxes_data.append({
                                    "position": {
                                        "minX": float(min_x)/224,
                                        "minY": float(min_y)/224,
                                        "maxX": float(max_x)/224,
                                        "maxY": float(max_y)/224
                                    },
                                    "class_id": class_id,
                                    "box_caption": f"Pred: {class_id}, Conf: {confidence:.2f}"
                                })
                            if max_x > min_x and max_y > min_y and conf[i][j].item()>0.2:
                                confidence = conf[i][j].item()
                                class_id = int(labels[i][j].argmax().item())
                                pred_boxes_data1.append({
                                    "position": {
                                        "minX": float(min_x)/224,
                                        "minY": float(min_y)/224,
                                        "maxX": float(max_x)/224,
                                        "maxY": float(max_y)/224
                                    },
                                    "class_id": class_id,
                                    "box_caption": f"Pred: {class_id}, Conf: {confidence:.2f}"
                                })

                    # Create prediction image
                    class_labels = {j: str(j) for j in range(91)}  # COCO dataset has 91 classes

                    make_box_arg = {
                        "predictions": {
                            "box_data": pred_boxes_data,
                            "class_labels": class_labels,
                        }
                    }
                    make_box_arg1 = {
                        "predictions": {
                            "box_data": pred_boxes_data1,
                            "class_labels": class_labels,
                        }
                    }

                    if pred_boxes_data:

                        wandb_pred_image = wandb.Image(img, boxes=make_box_arg)
                        wandb.log({f"pred_image_{i}": wandb_pred_image})
                    else:
                        # Log image without boxes if none are valid
                        wandb.log({f"pred_image_{i}_no_boxes": wandb.Image(img)})
                        print(f"Warning: No valid prediction boxes for image {i}")


                    if pred_boxes_data1:

                        wandb_pred_image = wandb.Image(img, boxes=make_box_arg1)
                        wandb.log({f"pred_image_{i}": wandb_pred_image})
                    else:
                        # Log image without boxes if none are valid
                        wandb.log({f"pred_image_{i}_no_boxes": wandb.Image(img)})
                        print(f"Warning: No valid prediction boxes for image {i}")

                    # GROUND TRUTH BOXES
                    gt_boxes_data = []
                    if len(box_real) > i and box_real[i] is not None:
                        for j, bbox in enumerate(box_real[i]):
                            if torch.sum(bbox).item() == 0:  # Skip padding boxes
                                continue

                            # Get coordinates - assuming [x_center, y_center, width, height] normalized format
                            x_center, y_center, width, height = bbox.tolist()

                            # Convert to absolute coordinates
                            x_center_abs = x_center * img_width
                            y_center_abs = y_center * img_height
                            width_abs = width * img_width
                            height_abs = height * img_height

                            # Calculate corners for wandb format
                            min_x = max(0, x_center_abs - (width_abs / 2))
                            min_y = max(0, y_center_abs - (height_abs / 2))
                            max_x = min(img_width, x_center_abs + (width_abs / 2))
                            max_y = min(img_height, y_center_abs + (height_abs / 2))

                            # Check for valid box
                            if max_x > min_x and max_y > min_y:
                                class_id = int(label_real[i][j].item()) if j < len(label_real[i]) else 0
                                gt_boxes_data.append({
                                    "position": {
                                        "minX": float(min_x)/224,
                                        "minY": float(min_y)/224,
                                        "maxX": float(max_x)/224,
                                        "maxY": float(max_y)/224
                                    },
                                    "class_id": class_id,
                                    "box_caption": f"GT: {class_id}"
                                })

                    # Create ground truth image
                    if gt_boxes_data:
                        make_gt_box = {
                            "ground_truth":{
                                "box_data": gt_boxes_data,
                                "class_labels": class_labels
                            }
                        }
                        wandb_gt_image = wandb.Image(img, boxes=make_gt_box)
                        wandb.log({f"gt_image_{i}": wandb_gt_image})
                    else:
                        # Log image without boxes if none are valid
                        wandb.log({f"gt_image_{i}_no_boxes": wandb.Image(img)})
                        print(f"Warning: No valid ground truth boxes for image {i}")

            return float(loss)

    def train(
            self,
            client_data: List[Tuple[torch.Tensor, torch.Tensor]],
            client_labels: List[Tuple[torch.Tensor, torch.Tensor]],
            n_epochs: int,
            batch_size: int,
            gradient_threshold: float = 1e-3,
            scheduler_type: str = "step",  # Options: "step", "cosine", "reduce_on_plateau", "exponential"
            scheduler_params: Dict = None,  # Parameters for the scheduler
            save_dir: str = "./saved_models"  # Directory to save models
        ) -> Dict:
            best_mse = np.inf
            best_epoch = -1
            history = []

            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Set default scheduler parameters if none provided
            if scheduler_params is None:
                scheduler_params = {}

            # Create schedulers directly inside the train function
            # StepLR scheduler (default)
            if scheduler_type == "step":
                step_size = scheduler_params.get("step_size", 20)
                gamma = scheduler_params.get("gamma", 0.75)
                top_scheduler = torch.optim.lr_scheduler.StepLR(self.top_optimizer, step_size=step_size, gamma=gamma)
                client_schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
                                    for opt in self.client_optimizers]

            # CosineAnnealingLR scheduler
            elif scheduler_type == "cosine":
                T_max = scheduler_params.get("T_max", n_epochs)  # Default to number of epochs
                eta_min = scheduler_params.get("eta_min", 0)
                top_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.top_optimizer, T_max=T_max, eta_min=eta_min)
                client_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
                                    for opt in self.client_optimizers]

            # ReduceLROnPlateau scheduler
            elif scheduler_type == "reduce_on_plateau":
                patience = scheduler_params.get("patience", 3)
                factor = scheduler_params.get("factor", 0.1)
                min_lr = scheduler_params.get("min_lr", 1e-6)
                top_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.top_optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True
                )
                client_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True
                ) for opt in self.client_optimizers]

            # ExponentialLR scheduler
            elif scheduler_type == "exponential":
                gamma = scheduler_params.get("gamma", 0.9)
                top_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.top_optimizer, gamma=gamma)
                client_schedulers = [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
                                    for opt in self.client_optimizers]

            # OneCycleLR scheduler
            elif scheduler_type == "one_cycle":
                # For OneCycleLR, we need to estimate total steps
                steps_per_epoch = min(len(data[0]) for data in client_data) // batch_size
                total_steps = n_epochs * steps_per_epoch
                max_lr = scheduler_params.get("max_lr", 0.01)

                top_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.top_optimizer, max_lr=max_lr, total_steps=total_steps
                )
                client_schedulers = [torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=max_lr, total_steps=total_steps
                ) for opt in self.client_optimizers]

            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

            for epoch in range(n_epochs):
                total_loss = 0.0
                c1 = 0.0
                b1 = 0.0
                cf1 = 0.0

                # Track only epoch-level gradient statistics for alerts
                epoch_grad_stats = {
                    'top_model': 0.0,
                    'client_models': [0.0 for _ in range(len(self.client_models))]
                }

                min_train_length = min(len(data[0]) for data in client_data)
                batch_start = torch.arange(0, min_train_length, batch_size)

                with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                    bar.set_description(f"Epoch {epoch}")
                    batch_count = 0

                    for start in bar:
                        client_batch_data = [
                            train_data[start:start+batch_size]
                            for train_data, _ in client_data
                        ]
                        client_batch_labels = [
                            train_labels[start:start+batch_size]
                            for train_labels, _ in client_labels
                        ]

                        # Forward and backward pass
                        loss, cl, bl, cfl = self.train_step(client_batch_data, client_batch_labels)

                        # Calculate top model gradient norm for this batch
                        top_model_grad_norm = 0.0
                        for name, param in self.top_model.named_parameters():
                            if param.grad is not None:
                                top_model_grad_norm += param.grad.data.norm(2).item()

                        # Accumulate top model gradient
                        epoch_grad_stats['top_model'] += top_model_grad_norm

                        # Calculate client models gradient norms for this batch
                        for i, model in enumerate(self.client_models):
                            client_grad_norm = 0.0
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    client_grad_norm += param.grad.data.norm(2).item()

                            # Accumulate client model gradient
                            epoch_grad_stats['client_models'][i] += client_grad_norm

                        # Perform optimizer steps
                        for optimizer in self.client_optimizers:
                            optimizer.step()
                        self.top_optimizer.step()

                        # Reset gradients
                        for optimizer in self.client_optimizers:
                            optimizer.zero_grad()
                        self.top_optimizer.zero_grad()

                        # Update training metrics
                        c1 += cl
                        b1 += bl
                        cf1 += cfl
                        total_loss += loss
                        bar.set_postfix(mse=float(loss))
                        batch_count += 1

                # Average the gradient norms across batches
                epoch_grad_stats['top_model'] /= batch_count
                for i in range(len(self.client_models)):
                    epoch_grad_stats['client_models'][i] /= batch_count

                # Calculate and log epoch-level statistics
                total_loss /= batch_count
                c1 /= batch_count
                b1 /= batch_count
                cf1 /= batch_count

                # Evaluate model
                min_test_length = min(len(test_data) for _, test_data in client_data)
                test_data = [test_data[:min_test_length] for _, test_data in client_data]
                mse = self.evaluate(test_data, client_labels[0][1][:min_test_length])
                rmse = math.sqrt(mse)

                if scheduler_type == "reduce_on_plateau":
                    top_scheduler.step(rmse)
                    for scheduler in client_schedulers:
                        scheduler.step(rmse)
                else:
                    top_scheduler.step()
                    for scheduler in client_schedulers:
                        scheduler.step()

                # Get current learning rates
                current_top_lr = self.top_optimizer.param_groups[0]['lr']
                current_client_lrs = [opt.param_groups[0]['lr'] for opt in self.client_optimizers]

                # Log learning rates to wandb
                wandb.log({"top_model_lr": current_top_lr})
                for i, lr in enumerate(current_client_lrs):
                    wandb.log({f"client_{i}_lr": lr})

                # Log basic metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": total_loss,
                    "class_loss": c1,
                    "box_loss": b1,
                    "conf_loss": cf1,
                    "top_model_grad_norm": epoch_grad_stats['top_model'],
                    "test_RMSE_loss": rmse
                })

                # Log client model gradients as metrics only
                for i, client_grad in enumerate(epoch_grad_stats['client_models']):
                    wandb.log({f"client_{i}_grad_norm": client_grad})

                # Check for vanishing gradients in top model
                if epoch_grad_stats['top_model'] < gradient_threshold:
                    warning_msg = f"WARNING: Potential vanishing gradient in top model (grad norm: {epoch_grad_stats['top_model']:.6f})"
                    print(warning_msg)
                    wandb.log({"top_model_vanishing_gradient_alert": 1})

                # Check for vanishing gradients in client models
                for i, client_grad in enumerate(epoch_grad_stats['client_models']):
                    if client_grad < gradient_threshold:
                        warning_msg = f"WARNING: Potential vanishing gradient in client model {i} (grad norm: {client_grad:.6f})"
                        print(warning_msg)
                        wandb.log({f"client_{i}_vanishing_gradient_alert": 1})

                # Log training metrics
                print(f"Epoch {epoch}, Average MSE: {total_loss}, Test RMSE: {rmse}, LR: {current_top_lr}")
                history.append(mse)

                # Only track the best epoch
                if mse < best_mse:
                    best_mse = mse
                    best_epoch = epoch
                    print(f"New best model performance at epoch {epoch} with MSE: {mse}")

            # Save final model weights
            timestamp = 1

            # Save top model weights
            top_model_path = os.path.join(save_dir, f"top_model_final_{timestamp}.pt")
            torch.save(self.top_model.state_dict(), top_model_path)
            print(f"Saved final top model weights to {top_model_path}")

            # Save client model weights
            client_model_paths = []
            for i, client_model in enumerate(self.client_models):
                client_model_path = os.path.join(save_dir, f"client_model_{i}_final_{timestamp}.pt")
                torch.save(client_model.state_dict(), client_model_path)
                client_model_paths.append(client_model_path)
                print(f"Saved final client model {i} weights to {client_model_path}")

            wandb.log({
                "model_saved": True,
                "top_model_path": top_model_path
            })
            
            mse = self.evaluate(test_data, client_labels[0][1][:min_test_length],save=True)

            return {
                'best_mse': best_mse,
                'best_epoch': best_epoch,
                'history': history,
                'top_model_path': top_model_path,
                'client_model_paths': client_model_paths
            }
