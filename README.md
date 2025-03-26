# Mixup-VFL: Leveraging Unaligned Data for Enhanced Vertical Federated Learning

This repository contains the implementation of Mixup-VFL, a framework that exploits all available data through innovative label-mixing strategies in Vertical Federated Learning (VFL) settings. The framework is designed to work effectively even when data is only partially aligned across participating entities, eliminating the need for entity alignment protocols like Private Set Intersection (PSI).

## Overview

Vertical Federated Learning enables privacy-preserving machine learning across distributed data sources but faces challenges when entities are only partially aligned across participants. Traditional approaches rely on identifying common entities, requiring computational overhead and focusing primarily on the aligned portion of data.

Mixup-VFL tackles this problem by utilizing innovative label-mixing strategies to leverage all available data, including unaligned portions. This approach significantly improves model performance, especially in scenarios with minimal data overlap.

## Key Features

- Multiple mixup strategies for regression tasks
- Object detection implementation within VFL paradigm
- Support for both aligned and unaligned data settings
- Comprehensive evaluation framework

## Mixup Strategies

The framework implements several label-mixing strategies:

1. **Max Mixup** - Selects the highest value among client labels
2. **Mean Mixup** - Computes the average of labels across clients
3. **Client-Importance Mixup** - Assigns weights based on statistical correlation
4. **Model-Based Importance Mixup** - Utilizes model-derived coefficients
5. **Mutual-Info Mixup** - Computes weights using mutual information

## Supported Tasks

### Regression
- Concrete Compressive Strength prediction
- Energy Efficiency prediction
- Real Estate Valuation
- Superconductivity prediction
- Yacht Hydrodynamics

### Object Detection
- Support for COCO and PASCAL VOC datasets
- Custom loss functions for bounding box regression

## Project Structure

```
mixup-vfl/
├── config/
│   ├── config.py               # Configuration enums and settings
│   └── vfl_config.yaml         # Example configuration file
├── data/
│   ├── regression_datasets.py  # Dataset loaders for regression tasks
│   ├── pascal_voc.py           # Pascal VOC dataset loader
│   └── utils.py                # Data processing utilities
├── mixup_vfl/
│   └── mixup_vfl.py            # Core implementation of Mixup-VFL
├── models/
│   └── various model implementations
├── utils/
│   ├── loss.py                 # Custom loss functions
│   └── mixup_strategies.py     # Implementation of mixup strategies
├── run_object_detection.py     # Script to run object detection experiments
└── run_regression.py           # Script to run regression experiments
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)
- Additional dependencies listed in requirements.txt

### Installation

```bash
git clone https://github.com/sachishs-15/Mixup-VFL.git
cd Mixup-VFL
pip install -r requirements.txt
```

### Running Regression Experiments

```bash
python run_regression.py --config config/vfl_config.yaml
```

### Running Object Detection Experiments

```bash
python run_object_detection.py --unaligned --mixup_strategy part_mixup --unaligned_ratio 0.8
```

## Configuration

The system is highly configurable through YAML configuration files. An example configuration:

```yaml
dataset:
  name: concrete
  subset_size: 1000
  train_test_ratio: 0.8
alignment:
  type: unaligned
  unalignment_ratio: 0.2
model:
  num_clients: 2
  embedding_size: 16
  mixup_strategy: mutual_info_mixup
  client_models:
  - hidden_layers:
    - 8
    - 16
    learning_rate: 0.001
    model_type: mlp
  - hidden_layers:
    - 8
    - 16
    learning_rate: 0.001
    model_type: mlp
  top_model:
    hidden_layers:
    - 32
    - 16
    learning_rate: 0.001
training:
  n_epochs: 100
  batch_size: 8
  device: cpu
  wandb:
    enabled: false
    project: vfl-project
    entity: vfl-team
    name: experiment-1
```

## Results

For regression tasks with only 20% aligned data, our mixup strategies achieved RMSE reductions of up to 72% compared to baseline methods. Similarly, our object detection implementation on COCO and PASCAL VOC datasets showed consistent improvements in test loss metrics when incorporating unaligned data.

## Acknowledgments

This research was supported by the Autonomous Ground Vehicle: Autonomy Intelligence (AGV.AI) Research Group at the Indian Institute of Technology Kharagpur.

## Citation

If you use this code in your research, please cite our paper:

```
@article{gudla2023mixupvfl,
  title={Mixup-VFL: Leveraging Unaligned Data for Enhanced Regression in Vertical Federated Learning},
  author={Gudla, Prudhvi and Singla, Sachish and Kumar, Ayush and Chakravarty, Devodita and Thirupathy, Varun and Amalanshu, Avi and Chakravarty, Debashish},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.