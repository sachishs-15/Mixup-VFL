#!/usr/bin/env python3
import argparse
import yaml
import os
import sys

# Default configuration template
DEFAULT_CONFIG = {
    "dataset": {
        "name": "california_housing",
        "subset_size": None,
        "train_test_ratio": 0.8
    },
    "alignment": {
        "type": "aligned",
        "unalignment_ratio": 0.8
    },
    "model": {
        "num_clients": 2,
        "embedding_size": 8,
        "mixup_strategy": "no_mixup",
        "client_models": [
            {
                "hidden_layers": [12, 6],
                "learning_rate": 0.001,
                "model_type": "mlp"
            },
            {
                "hidden_layers": [12, 6],
                "learning_rate": 0.001,
                "model_type": "mlp"
            }
        ],
        "top_model": {
            "hidden_layers": [24, 12],
            "learning_rate": 0.001,
            "aggregate_fn": None
        }
    },
    "training": {
        "n_epochs": 100,
        "batch_size": 64,
        "device": "cuda",
        "wandb": {
            "enabled": True,
            "project": "VFL-Regression",
            "entity": "vfl",
            "name": "experiment"
        }
    },
    "feature_distribution": None
}

AVAILABLE_DATASETS = [
    "california_housing",
    "wine",
    "mining_process",
    "biketrip",
    "superconductivity",
    "ice_pets"
]

ALIGNMENT_TYPES = ["aligned", "unaligned"]

MIXUP_STRATEGIES = [
    "no_mixup",
    "max_mixup",
    "mean_mixup",
    "importance_mixup",
    "model_based_mixup",
    "mutual_info_mixup"
]

def create_config_file(args):
    """Create a config file with the specified parameters"""
    config = DEFAULT_CONFIG.copy()
    
    # Update dataset parameters
    if args.dataset:
        if args.dataset not in AVAILABLE_DATASETS:
            print(f"Warning: {args.dataset} not in known datasets: {AVAILABLE_DATASETS}")
        config["dataset"]["name"] = args.dataset
    
    if args.subset_size is not None:
        config["dataset"]["subset_size"] = args.subset_size
    
    if args.train_test_ratio is not None:
        config["dataset"]["train_test_ratio"] = args.train_test_ratio
    
    # Update alignment parameters
    if args.alignment:
        if args.alignment not in ALIGNMENT_TYPES:
            print(f"Warning: {args.alignment} not in known alignment types: {ALIGNMENT_TYPES}")
        config["alignment"]["type"] = args.alignment
    
    if args.unalignment_ratio is not None:
        config["alignment"]["unalignment_ratio"] = args.unalignment_ratio
    
    # Update model parameters
    if args.num_clients is not None:
        config["model"]["num_clients"] = args.num_clients
        # Adjust client_models array to match num_clients
        if len(config["model"]["client_models"]) < args.num_clients:
            default_client = {
                "hidden_layers": [12, 6],
                "learning_rate": 0.001,
                "model_type": "mlp"
            }
            for _ in range(args.num_clients - len(config["model"]["client_models"])):
                config["model"]["client_models"].append(default_client.copy())
        elif len(config["model"]["client_models"]) > args.num_clients:
            config["model"]["client_models"] = config["model"]["client_models"][:args.num_clients]
    
    if args.embedding_size is not None:
        config["model"]["embedding_size"] = args.embedding_size
    
    if args.mixup_strategy:
        if args.mixup_strategy not in MIXUP_STRATEGIES:
            print(f"Warning: {args.mixup_strategy} not in known mixup strategies: {MIXUP_STRATEGIES}")
        config["model"]["mixup_strategy"] = args.mixup_strategy
    
    # Update training parameters
    if args.epochs is not None:
        config["training"]["n_epochs"] = args.epochs
    
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    
    if args.device:
        config["training"]["device"] = args.device
    
    if args.no_wandb:
        config["training"]["wandb"]["enabled"] = False
    
    # Write the config to file
    output_file = args.output
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration file created at: {output_file}")

def main():
    """Parse arguments and create config file"""
    parser = argparse.ArgumentParser(description='Create a configuration file for VFL training')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=AVAILABLE_DATASETS,
                        help=f'Dataset to use. Options: {", ".join(AVAILABLE_DATASETS)}')
    parser.add_argument('--subset-size', type=int, help='Number of samples to use (subset of the dataset)')
    parser.add_argument('--train-test-ratio', type=float, help='Ratio of training to testing data (0.8 = 80% train)')
    
    # Alignment parameters
    parser.add_argument('--alignment', type=str, choices=ALIGNMENT_TYPES,
                        help=f'Data alignment type. Options: {", ".join(ALIGNMENT_TYPES)}')
    parser.add_argument('--unalignment-ratio', type=float, help='Ratio of unaligned data')
    
    # Model parameters
    parser.add_argument('--num-clients', type=int, help='Number of client models')
    parser.add_argument('--embedding-size', type=int, help='Size of embeddings from client models')
    parser.add_argument('--mixup-strategy', type=str, choices=MIXUP_STRATEGIES,
                        help=f'Mixup strategy to use. Options: {", ".join(MIXUP_STRATEGIES)}')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    
    # Output file
    parser.add_argument('--output', type=str, default='vfl_config.yaml', help='Output config file path')
    
    args = parser.parse_args()
    create_config_file(args)

if __name__ == "__main__":
    main()