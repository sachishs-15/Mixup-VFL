#!/usr/bin/env python3
import argparse
import yaml
import random
import numpy as np
import torch
import math
import sys
from typing import Dict, List, Tuple, Optional
import wandb
import psycopg2
from psycopg2 import sql
from datetime import datetime
from pdb import set_trace

DB_HOST = "yamanote.proxy.rlwy.net"
DB_PORT = 42901
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASSWORD = "TvROSyEFxjKowwovGUSBuLHumfmhzuck"

conn = None

def get_db_connection():
    """Get or create a database connection"""
    global conn
    if conn is None or conn.closed:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            conn.autocommit = False  # We'll manage transactions explicitly
            print("✅ Successfully connected to Railway PostgreSQL database")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return None
    return conn

def insert_ml_config(config_data):
    """Insert a record into the ML experiment configurations table"""
    global conn
    conn = get_db_connection()
    if not conn:
        return False
    
    cur = conn.cursor()
    try:
        # Prepare the column names and values
        columns = ', '.join(config_data.keys())
        placeholders = ', '.join(['%s'] * len(config_data))

        # set_trace()
        
        # Build the INSERT statement
        insert_query = f'''
        INSERT INTO results ({columns})
        VALUES ({placeholders})
        RETURNING id
        '''
        
        # Execute the query with the values
        cur.execute(insert_query, list(config_data.values()))
        
        # Get the ID of the new record
        new_id = cur.fetchone()[0]
        
        conn.commit()
        print(f"✅ Configuration inserted with ID: {new_id}")
        return new_id
    except Exception as e:
        conn.rollback()
        print(f"❌ Error inserting data: {e}")
        return None
    finally:
        cur.close()

# Import the custom VFL implementation
from vfl_implementation import (
    CustomizableVFL,
    DataAlignment,
    MixupStrategy,
    split_features,
    california_housing,
    wine,
    mining_process,
    biketrip,
    superconductivity,
    ice_pets
)

def get_dataset(dataset_name):
    """Get dataset based on name from config"""
    if dataset_name == "california_housing":
        return california_housing()
    elif dataset_name == "wine":
        return wine()
    elif dataset_name == "mining_process":
        return mining_process()
    elif dataset_name == "biketrip":
        return biketrip()
    elif dataset_name == "superconductivity":
        return superconductivity()
    elif dataset_name == "ice_pets":
        return ice_pets()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def run_program(config_path):
    """Run VFL system with the given config file"""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract configuration parameters
    dataset_config = config['dataset']
    alignment_config = config['alignment']
    model_config = config['model']
    training_config = config['training']
    
    # Set data alignment
    data_alignment = DataAlignment(alignment_config['type'])
    unalignment_ratio = alignment_config['unalignment_ratio']
    
    # Set mixup strategy
    mixup_strategy = MixupStrategy(model_config['mixup_strategy'])
    
    print(f"Dataset: {dataset_config['name']}")
    print(f"Mixup Strategy: {mixup_strategy.value}")
    print(f"Data Alignment: {data_alignment.value}")
    
    # Initialize WandB if enabled
    if training_config.get('wandb', {}).get('enabled', False):
        wandb_config = training_config['wandb']
        wandb.init(
            project=wandb_config.get('project', 'VFL-Regression'),
            entity=wandb_config.get('entity', 'vfl'),
            name=wandb_config.get('name', 'experiment'),
            config=config
        )
    
    # Load data
    X, y, num_features = get_dataset(dataset_config['name'])
    
    # Apply subset size if specified
    subset_size = dataset_config.get('subset_size')
    if subset_size is not None:
        print(f"Using subset of {subset_size} samples")
        X = X[:subset_size]
        y = y[:subset_size]
    
    print(f"Data loaded: {X.shape} features, {y.shape} labels")
    
    # Create feature splits
    feature_distribution = config.get('feature_distribution')
    num_clients = model_config['num_clients']
    feature_splits = split_features(
        num_features=num_features, 
        num_clients=num_clients, 
        distribution=feature_distribution
    )
    
    print(f"Feature splits: {feature_splits}")
    
    # Configure device
    device = training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize VFL system
    vfl = CustomizableVFL(
        num_clients=num_clients,
        feature_splits=feature_splits,
        data_alignment=data_alignment,
        client_models_config=model_config['client_models'],
        top_model_config=model_config['top_model'],
        embedding_size=model_config['embedding_size'],
        mixup_strategy=mixup_strategy,
        device=device
    )
    
    print("VFL system initialized")
    
    # Prepare datasets
    client_data, client_labels = vfl.prepare_datasets(
        X=X, 
        y=y, 
        subset_size=None,  # Already applied earlier
        train_size=dataset_config['train_test_ratio'],
        unaligned_ratio=unalignment_ratio
    )
    
    print("Data prepared for client models")
    
    # Train the system
    results = vfl.train(
        client_data=client_data,
        client_labels=client_labels,
        n_epochs=training_config['n_epochs'],
        batch_size=training_config['batch_size']
    )
    
    print(f"Training completed. Final Best MSE: {results['best_mse']}, RMSE: {math.sqrt(results['best_mse'])}")

    # Insert results into the database
    res = {
       
        "dataset": dataset_config['name'],
        "subset_size": dataset_config['subset_size'],
        "train_test_ratio":  dataset_config['train_test_ratio'],
        "aligned": True if data_alignment == DataAlignment.ALIGNED else False,
        "unaligned_ratio": alignment_config['unalignment_ratio'] if data_alignment == DataAlignment.UNALIGNED else 0,
        "num_clients":  model_config['num_clients'],
        "embedding_size": model_config['embedding_size'],
        "mixup_strategy": model_config['mixup_strategy'],
        "epochs": training_config['n_epochs'],
        "batch_size": training_config['batch_size'],
        "best_mse": results['best_mse'],
    }

    insert_ml_config(res)

    return results

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run VFL with config file')
    parser.add_argument('--config', type=str, default='vfl_config.yaml', help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run the program
    run_program(args.config)

if __name__ == "__main__":
    
    main()