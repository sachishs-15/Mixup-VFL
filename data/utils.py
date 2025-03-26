from typing import List, Tuple, Optional

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