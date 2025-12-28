"""
Mental Health Prediction Framework

A modular machine learning framework for predicting mental health disorder prevalence.

Modules:
- config: Configuration and parameters
- data_loader: Data loading and preprocessing
- feature_engineering: Feature creation and transformation
- clustering: Clustering analysis and evaluation
- ml_models: Machine learning models and training
- visualization: Plotting and figure generation
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import config
from . import data_loader
from . import feature_engineering
from . import clustering
from . import ml_models
from . import visualization

__all__ = [
    'config',
    'data_loader',
    'feature_engineering',
    'clustering',
    'ml_models',
    'visualization'
]
