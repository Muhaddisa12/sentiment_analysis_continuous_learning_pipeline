"""
Configuration file for Self-Training Sentiment Analysis System.
Contains all system parameters, thresholds, and paths.
"""

import os

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "sentiment_analysis"
}

# Model Storage Paths
MODEL_DIR = "models_store"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"
CURRENT_MODEL_PATH = f"{MODEL_DIR}/current_model.pkl"
VECTORIZER_PATH = f"{MODEL_DIR}/vectorizer.pkl"
EXPERIMENTS_DIR = "experiments"

# Drift Detection Configuration
DRIFT_THRESHOLD = 0.3  # Overall drift score threshold for triggering retraining
FEATURE_DRIFT_WEIGHT = 0.6  # Weight for feature distribution drift
CONFIDENCE_DRIFT_WEIGHT = 0.4  # Weight for confidence drift

# Self-Training Configuration
CONFIDENCE_THRESHOLD = 0.85  # Minimum prediction probability to accept sample for self-training
LOW_CONFIDENCE_DIR = "low_confidence_samples"  # Directory for storing low-confidence samples

# Model Lifecycle Configuration
MIN_IMPROVEMENT_THRESHOLD = 0.01  # Minimum accuracy improvement to deploy new model (1%)
ROLLBACK_ON_DEGRADATION = True  # Automatically rollback if new model performs worse

# Experiment Mode Configuration
EXPERIMENT_MODE = True  # Enable research experiment tracking
METRICS_LOG_PATH = f"{EXPERIMENTS_DIR}/metrics_history.json"
DRIFT_LOG_PATH = f"{EXPERIMENTS_DIR}/drift_history.json"

# Logging Configuration
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/system.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOW_CONFIDENCE_DIR, exist_ok=True)
