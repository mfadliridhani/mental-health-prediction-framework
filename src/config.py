"""
Configuration file for mental health prediction framework.
Contains constants, paths, and parameters used throughout the project.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Data files
DATA_FILES = {
    'prevalence': '1- mental-illnesses-prevalence.csv',
    'coverage': '4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv',
    'symptoms': '6- depressive-symptoms-across-us-population.csv',
    'countries': '7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv'
}

# Feature column mappings
COLUMN_MAPPING = {
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depression',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating'
}

# Target variable
TARGET_VARIABLE = 'Eating'

# Base mental health features
BASE_FEATURES = ['Schizophrenia', 'Depression', 'Anxiety', 'Bipolar', 'Eating']

# Columns to exclude from modeling
EXCLUDE_COLUMNS = ['Entity', 'Code', 'Year', 'Eating', 
                   'KMeans_Cluster', 'DBSCAN_Cluster', 'Hierarchical_Cluster',
                   'PCA1', 'PCA2', 'TSNE1', 'TSNE2']

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Clustering parameters
OPTIMAL_K = 5
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
PCA_COMPONENTS = 2
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 5,
        'random_state': RANDOM_STATE
    },
    'svr': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'neural_network': {
        'hidden_layer_sizes': (100, 50, 25),
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'early_stopping': True
    },
    'ridge': {
        'alpha': 1.0
    },
    'lasso': {
        'alpha': 0.1
    },
    'elastic_net': {
        'alpha': 0.1,
        'l1_ratio': 0.5
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
}

# Visualization settings
PLOT_WIDTH = 900
PLOT_HEIGHT = 600
DPI = 150
COLOR_PALETTE = 'viridis'
