from typing import List
import torch
import numpy as np
from sklearn.linear_model import LinearRegression   
from sklearn.feature_selection import mutual_info_regression


def no_mixup(
    client_batch_labels: List[torch.Tensor],
    client_embeddings: List[torch.Tensor] = None
) -> torch.Tensor:
    return client_batch_labels[0]
    
def max_mixup(
    client_batch_labels: List[torch.Tensor],
    client_embeddings: List[torch.Tensor] = None
) -> torch.Tensor:
    return torch.max(torch.stack(client_batch_labels), dim=0).values
    
def mean_mixup(
    client_batch_labels: List[torch.Tensor],
    client_embeddings: List[torch.Tensor] = None
) -> torch.Tensor:
    return torch.mean(torch.stack(client_batch_labels), dim=0)

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
    
def model_based_mixup(
    client_batch_labels: List[torch.Tensor],
    client_embeddings: List[torch.Tensor],
    model_type: str = 'linear'
) -> torch.Tensor:
    
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
    
def mutual_info_mixup(
    client_batch_labels: List[torch.Tensor],
    client_embeddings: List[torch.Tensor]
) -> torch.Tensor:
        
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