"""
Emotion Classification Research Package
"""

from .config import config
from .data_loader import load_bigru_data, load_distilbert_data
from .train_bigru import train_bigru, BiGRUClassifier
from .train_distilbert import train_distilbert
from .evaluate import evaluate_models
from .deploy import app

__version__ = "1.0.0"
__all__ = [
    "config",
    "load_bigru_data",
    "load_distilbert_data",
    "train_bigru",
    "BiGRUClassifier",
    "train_distilbert",
    "evaluate_models",
    "app",
]

print("src package loaded successfully!")


