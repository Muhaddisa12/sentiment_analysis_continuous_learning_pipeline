"""
Machine Learning module for sentiment analysis.
Contains models, training, evaluation, preprocessing, and drift detection.
"""

from .models import MODELS
from .trainer import train_pipeline
from .evaluator import evaluate
from .preprocessing import create_vectorizer
from .data_loader import load_database

__all__ = [
    'MODELS',
    'train_pipeline',
    'evaluate',
    'create_vectorizer',
    'load_database'
]
