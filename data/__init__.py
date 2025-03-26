from .coco import load_coco_data
from .pascal_voc import load_pascal_data
from .regression_datasets import real_estate, concrete, energy, yacht_hydrodynamics, superconductivity

__all__ = [
    "load_coco_data",
    "load_pascal_data",
    "real_estate",
    "concrete",
    "energy",
    "yacht_hydrodynamics",
    "superconductivity",
    "get_dataset"
]