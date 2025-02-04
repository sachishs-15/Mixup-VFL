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
import argparse
from sklearn.datasets import fetch_california_housing
import random


class DataAlignment(Enum):
    ALIGNED = "aligned"
    UNALIGNED = "unaligned"

class MixupStrategy(Enum):
    NO_MIXUP = "no_mixup"
    MAX_MIXUP = "max_mixup"
    MEAN_MIXUP = "mean_mixup"

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

class ServerModel(nn.Module):
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
        self.top_model = ServerModel(
            num_clients=num_clients,
            embedding_size=embedding_size,
            hidden_layers=top_model_config['hidden_layers']
        ).to(device)
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
        
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def no_mixup(
        client_batch_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        return client_batch_labels[0]
    
    @staticmethod
    def max_mixup(
        client_batch_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        return torch.max(torch.stack(client_batch_labels), dim=0).values
    
    @staticmethod
    def mean_mixup(
        client_batch_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        return torch.mean(torch.stack(client_batch_labels), dim=0)
    
    def prepare_datasets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        unaligned_ratio: float = 0.8
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Same as before - preparation of aligned or unaligned datasets"""
        client_data = []
        client_labels = []
        
        if self.data_alignment == DataAlignment.ALIGNED:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=True
            )
            
            for feature_split in self.feature_splits:
                X_train = torch.tensor(X_train_full[:, feature_split], dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test_full[:, feature_split], dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_tensor, y_test_tensor))

        elif self.data_alignment == DataAlignment.UNALIGNED:

            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=True
            )

            # pdb.set_trace()
            all_indices_random = np.random.permutation(len(X_train_full))
            unaligned_indices = all_indices_random[:int(len(X_train_full) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X_train_full)), unaligned_indices)
            
            for feature_split in self.feature_splits:

                train_unaligned_indices = unaligned_indices.copy()
                np.random.shuffle(unaligned_indices)
                train_aligned_indices = aligned_indices
                client_indices = np.concatenate([train_aligned_indices, train_unaligned_indices]) # concatenate aligned and unaligned indices

                X_train = torch.tensor(X_train_full[:, feature_split], dtype=torch.float32).to(self.device)
                X_train = X_train[list(client_indices)]
                X_test = torch.tensor(X_test_full[:, feature_split], dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train[list(client_indices)], dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
                
                client_data.append((X_train, X_test))
                client_labels.append((y_train_tensor, y_test_tensor))

        else:

            all_indices_random = np.random.permutation(len(X))
            unaligned_indices = all_indices_random[:int(len(X) * unaligned_ratio)]
            aligned_indices = np.setdiff1d(np.arange(len(X)), unaligned_indices)

            for feature_split in self.feature_splits:
                X_client = X[:, feature_split]
                # client_unaligned_indices = np.random.permutation(len(X_client))
                # client_unaligned_indices = np.random.shuffle(client_unaligned_indices)
                # client_aligned_indices = aligned_indices
                # indices = np.concatenate([client_aligned_indices, client_unaligned_indices])
                # client_size = int(len(indices) * unaligned_ratio)
                # client_indices = indices[:client_size]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X[:, feature_split],
                    y,
                    train_size=train_size,
                    shuffle=False
                )
                
                X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
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
        loss = self.loss_fn(final_prediction, self.mixup_fn(client_batch_labels))
        
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
            # pdb.set_trace()
            min_test_length = min(len(test_data) for _, test_data in client_data)
            test_data = [test_data[:min_test_length] for _, test_data in client_data]
            mse = self.evaluate(test_data, client_labels[0][1][:min_test_length])
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

def run_program():

    # Parse the Arguments
    parser = argparse.ArgumentParser(description='Customize how you want to run VFL')
    parser.add_argument('--unaligned', action='store_true', help='Whether to run VFL with aligned data')
    parser.add_argument('--unaligned_ratio', type=float, default=0.8, help='Ratio of unaligned data')
    parser.add_argument('--mixup_strategy', type=str, default='no_mixup', help='Mixup strategy to use', choices=['no_mixup', 'max_mixup', 'mean_mixup'])
    
    args = parser.parse_args()

    algn_type = DataAlignment.ALIGNED if not args.unaligned else DataAlignment.UNALIGNED
    mixup_strategy = MixupStrategy(args.mixup_strategy)
    print(f"Mixup Strategy: {mixup_strategy}")
    print(f"Data Alignment: {algn_type}")
    
    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target

    print("data loaded")
    
    # Configuration
    num_clients = 3
    feature_splits = [
        [0, 1, 2],  # Client 1 features
        [3, 4, 5],  # Client 2 features
        [6, 7]      # Client 3 features
    ]
    
    # Client model configurations
    client_models_config = [
        {'hidden_layers': [12, 6], 'learning_rate': 0.001},
        {'hidden_layers': [12, 6], 'learning_rate': 0.001},
        {'hidden_layers': [8, 4], 'learning_rate': 0.001}
    ]

    print("client config loaded")

    # Top model configuration
    top_model_config = {
        'hidden_layers': [24, 12],
        'learning_rate': 0.001
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize VFL system
    vfl = CustomizableVFL(
        num_clients=num_clients,
        feature_splits=feature_splits,
        data_alignment=algn_type,
        client_models_config=client_models_config,
        top_model_config=top_model_config,
        embedding_size=8,
        mixup_strategy=mixup_strategy,
        device=device
    )

    print("VFL initialized")
    
    # Prepare datasets
    client_data, client_labels = vfl.prepare_datasets(X, y)

    print("data prepared")
    
    # Train the system
    results = vfl.train(
        client_data=client_data,
        client_labels=client_labels,
        n_epochs=100,
        batch_size=10
    )
    
    print(f"Final Best MSE: {results['best_mse']}")


seed = 42  # Choose any fixed number
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    run_program()