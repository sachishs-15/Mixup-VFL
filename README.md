# Mixup-VFL: Leveraging Unaligned Data for Enhanced Vertical Federated Learning

Vertical Federated Learning (VFL) enables privacy-preserving machine learning across distributed data sources but faces a critical challenge when entities are only partially aligned across participants. Traditional approaches rely on Private Set Intersection (PSI) to identify common entities, requiring computational overhead and focusing primarily on the aligned portion of data.

We propose Mixup-VFL, a framework that exploits all available data through innovative label-mixing strategies, eliminating the need for entity alignment protocols like PSI. We evaluate various mixing approaches for regression tasks and explore their applicability to object detection by reframing it as a regression problem within the VFL paradigm.

Extensive experiments across multiple datasets demonstrate that Mixup-VFL consistently outperforms entity-aligned-only approaches, with particularly substantial gains in scenarios with minimal data overlap. For regression tasks with only 20% aligned data, our mixup strategies achieved RMSE reductions up to 72% compared to baseline methods. Similarly, our object detection implementation on COCO and PASCAL VOC datasets showed consistent improvements in test loss metrics when incorporating unaligned data.

Our approach offers a practical solution for real-world VFL deployments where data alignment is limited but valuable information exists across all available data.

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

Support for various datasets:

- Concrete Compressive Strength 
- Energy Efficiency 
- Real Estate Valuation
- Superconductivity 
- Yacht Hydrodynamics

### Object Detection
- Support for COCO and PASCAL VOC datasets
- Custom loss functions for bounding box regression

## Project Structure

```
VFL-Regression/
├── config/
│   ├── config.py               # Configuration enums and settings
│   └── vfl_config.yaml         # Example configuration file
├── data/
│   ├── __init__.py
│   ├── coco.py                 # COCO dataset loader
│   ├── pascal_voc.py           # Pascal VOC dataset loader
│   ├── regression_datasets.py  # Dataset loaders for regression tasks
│   ├── utils.py                # Data processing utilities
│   └── Datasets/               # Directory for dataset storage
├── mixup_vfl/
│   └── mixup_vfl.py            # Core implementation of Mixup-VFL
├── models/
│   ├── __init__.py
│   ├── client_models.py        # Client-side model implementations
│   └── server_models.py        # Server-side model implementations
└── utils/
│   ├── loss.py                 # Custom loss functions
│   └── mixup_strategies.py     # Implementation of mixup strategies
├── run_object_detection.py     # Script to run object detection 
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