import numpy as np
from pdb import set_trace

def compute_client_importance(embeddings_dict, labels):
    """
    Compute client importance coefficients based on correlation between embeddings and labels.
    
    Args:
        embeddings_dict (dict): Dictionary where keys are client IDs and values are embedding matrices (numpy arrays).
        labels (numpy array): Target labels at the server.
        
    Returns:
        dict: Normalized client importance coefficients.
    """
    client_importance = {}  

    set_trace()
    
    for client_id, embeddings in embeddings_dict.items():
        correlations = []
        for i in range(embeddings.shape[1]):  # Iterate over embedding dimensions
            corr = np.corrcoef(embeddings[:, i], labels)[0, 1]
            correlations.append(abs(corr))  # Use absolute correlation
        
        # Aggregate correlations for this client
        client_importance[client_id] = np.mean(correlations)
    
    # Normalize importance scores to sum to 1
    total_importance = sum(client_importance.values())
    for client_id in client_importance:
        client_importance[client_id] /= total_importance
    
    return client_importance

# Example usage
embeddings_dict = {
    'client_1': np.random.rand(100, 5),  # Embeddings from Client 1
    'client_2': np.random.rand(100, 3),  # Embeddings from Client 2
}
labels = np.random.rand(100)  # Labels at the server
importance_scores = compute_client_importance(embeddings_dict, labels)
print(importance_scores)