import argparse
import yaml
import random
import numpy as np
import torch
import math
import sys
from typing import Dict, List, Tuple, Optional
import wandb
from pdb import set_trace
from data.regression_datasets import get_dataset
from config.config import DataAlignment, MixupStrategy
from data.utils import split_features
from mixup_vfl import MixupVFL_ObjectDetection
from data.pascal_voc import load_pascal_data

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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    client_models_config = [
        {'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600},
        {'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600},
        {'learning_rate': 0.001, 'model_type': 'resnet50', 'image_width': 600}
    ]

    print("client config loaded")

    # Top model configuration
    top_model_config = {
        'hidden_layers': [24, 12],
        'learning_rate': 0.001,
        'server':'object'
    }

    # Initialize VFL system
    vfl = MixupVFL_ObjectDetection(
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