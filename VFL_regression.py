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
from DataLoader import load_coco_data
from DataLoader import load_pascal_data
import torch.utils.checkpoint as cp
from Loss import object_detection_loss
import torch.nn.functional as F
import datetime
import Model as model
class DataAlignment(Enum):
    ALIGNED = "aligned"
    UNALIGNED = "unaligned"

class MixupStrategy(Enum):
    NO_MIXUP = "no_mixup"
    ADD_MIXUP="add_mixup"
    MEAN_MIXUP = "mean_mixup"
    IMPORTANCE_MIXUP = "importance_mixup"
    MODEL_BASED_MIXUP = "model_based_mixup"
    MUTUAL_INFO_MIXUP = "mutual_info_mixup"


cmm=[]
tp=[]
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
                model = model.EfficientNetClientModel(
                    embedding_size=embedding_size,
                    pretrained=config.get('pretrained', True)
                ).to(device)
            else:
                model = model.ClientModel(
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

        # if config.get('server')=='object':
        self.top_model = model.ResNetBBoxPredictor(
             num_clients=num_clients,
             total_embedding_size=num_clients*embedding_size,
             num_classes=91,
             num_boxes=20,
             conf_threshold=0.7
         ).to(device)
        tp=self.top_model
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
        elif self.mixup_strategy == MixupStrategy.ADD_MIXUP:
            self.mixup_fn = self.add_mixup
        elif self.mixup_strategy == MixupStrategy.PART_MIXUP:
            self.mixup_fn = self.part_mixup
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
        print("here")
        return client_batch_labels[0]

    @staticmethod
    def add_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combines batch labels from multiple clients by merging all bounding boxes and labels
        for corresponding image positions.

        Args:
            client_batch_labels: List of tensors containing batch labels from each client
            client_embeddings: Not used, kept for API compatibility

        Returns:
            torch.Tensor: Combined labels from all clients
        """
        num_clients = len(client_batch_labels)
        batch_size = len(client_batch_labels[0])
        combined_labels = []

        # For each batch position
        for batch_idx in range(batch_size):
            # Get all bounding boxes and labels for this image position from all clients
            all_boxes_and_labels = []

            for client_idx in range(num_clients):
                image_labels = client_batch_labels[client_idx][batch_idx]
                all_boxes_and_labels.append(image_labels)

            # Concatenate all boxes and labels for this image
            combined_image_labels = torch.cat(all_boxes_and_labels, dim=0)
            combined_labels.append(combined_image_labels)

        return combined_labels


    @staticmethod
    def part_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor] = None
    ) -> torch.Tensor:
         return client_batch_labels[0]



    @staticmethod
    def importance_mixup(
        client_batch_labels: List[torch.Tensor],
        client_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:

        #  ()
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

        #  ()
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
        truth = self.mixup_fn(client_batch_labels, concatenated_embeddings)

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

                # Step the schedulers - this adjusts the learning rate
                # For ReduceLROnPlateau, we need to provide the validation metric
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

            # Log to wandb
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
  #   ()
    df = df.drop(df.columns[0], axis=1)  # Drop the first column

    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.astype(float)

    data = df[df.columns[:-1]]
    target = df[df.columns[-1]]

    data = data.to_numpy()
    target = target.to_numpy()
    #  ()
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
    parser.add_argument('--mixup_strategy', type=str, default='add_mixup', help='Mixup strategy to use', choices=['no_mixup', 'add_mixup', 'part_mixup', 'importance_mixup', 'model_based_mixup', 'mutual_info_mixup'])

    args = parser.parse_args([])

    algn_type = DataAlignment.UNALIGNED if not args.unaligned else DataAlignment.ALIGNED
    mixup_strategy = MixupStrategy(args.mixup_strategy)
    unaligned_ratio = args.unaligned_ratio
    print(f"Mixup Strategy: {mixup_strategy}")
    print(f"Data Alignment: {algn_type}")

    # Load data
    X,y,X_test,y_test,feat_no= load_pascal_data(num_train=1000,num_val=200)
    print("data loaded")

    # Configuration
    num_clients = 2
    (c,h,w)=X[0].shape
    feat_no = w
    a = split_features(feat_no, num_clients)
    feature_splits=[]
    for i in a:
        feature_splits.append((c,h,i))

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
    vfl.Y_val=y_test
    vfl.X_val=X_test

    print("VFL initialized")


    client_data, client_labels = vfl.prepare_datasets(X, y,X_test,y_test,subset_size=250, train_size=0.8, unaligned_ratio=unaligned_ratio)


    print("data prepared")

    # Train the system
    wandb.init(project="coco_training", name="experiment_8_unaligned_20_pascal_eff_boxes", config={"epochs": 100})
    results = vfl.train(
        client_data=client_data,
        client_labels=client_labels,
        n_epochs=100,
        batch_size=16,
    )

    wandb.finish()

    print(f"Final Best MSE: {results['best_mse']}")


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    run_program()