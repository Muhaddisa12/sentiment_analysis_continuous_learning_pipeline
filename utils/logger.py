"""
Structured Logging System for MLOps Pipeline

This module provides centralized, structured logging for:
- Training events
- Drift detection
- Model deployments
- System errors

Logs are both human-readable and machine-parseable for analysis.
"""

import logging
import os
from datetime import datetime
from config import LOG_DIR, LOG_FILE, LOG_LEVEL


def setup_logger():
    """
    Configure and return the system logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('sentiment_analysis_system')
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler for persistent logging
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter with timestamps and structured information
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
_logger = None


def get_logger():
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log_training_start(model_name=None, dataset_size=None):
    """
    Log the start of a training pipeline.
    
    Args:
        model_name: Name of model being trained (optional)
        dataset_size: Size of training dataset (optional)
    """
    logger = get_logger()
    msg = "TRAINING_START"
    if model_name:
        msg += f" | model={model_name}"
    if dataset_size:
        msg += f" | dataset_size={dataset_size}"
    logger.info(msg)


def log_training_end(model_name, accuracy, training_time=None):
    """
    Log the completion of training.
    
    Args:
        model_name: Name of trained model
        accuracy: Validation accuracy achieved
        training_time: Time taken for training (seconds, optional)
    """
    logger = get_logger()
    msg = f"TRAINING_END | model={model_name} | accuracy={accuracy:.4f}"
    if training_time:
        msg += f" | duration={training_time:.2f}s"
    logger.info(msg)


def log_drift_detection(feature_drift, confidence_drift, overall_drift, triggered_retrain=False):
    """
    Log drift detection event.
    
    Args:
        feature_drift: Feature distribution drift score
        confidence_drift: Confidence drift score
        overall_drift: Overall combined drift score
        triggered_retrain: Whether drift triggered retraining
    """
    logger = get_logger()
    msg = (
        f"DRIFT_DETECTION | feature_drift={feature_drift:.4f} | "
        f"confidence_drift={confidence_drift:.4f} | "
        f"overall_drift={overall_drift:.4f} | "
        f"retrain_triggered={triggered_retrain}"
    )
    logger.info(msg)


def log_model_switch(old_version, new_version, reason, old_accuracy=None, new_accuracy=None):
    """
    Log model version switch (deployment or rollback).
    
    Args:
        old_version: Previous model version
        new_version: New model version
        reason: Reason for switch (e.g., "drift_retrain", "rollback")
        old_accuracy: Previous model accuracy (optional)
        new_accuracy: New model accuracy (optional)
    """
    logger = get_logger()
    msg = f"MODEL_SWITCH | from=v{old_version} | to=v{new_version} | reason={reason}"
    if old_accuracy is not None:
        msg += f" | old_accuracy={old_accuracy:.4f}"
    if new_accuracy is not None:
        msg += f" | new_accuracy={new_accuracy:.4f}"
    logger.info(msg)


def log_error(error_type, message, exception=None):
    """
    Log system errors.
    
    Args:
        error_type: Type of error (e.g., "TRAINING_ERROR", "DRIFT_ERROR")
        message: Error message
        exception: Exception object (optional)
    """
    logger = get_logger()
    msg = f"ERROR | type={error_type} | message={message}"
    if exception:
        msg += f" | exception={type(exception).__name__}"
    logger.error(msg, exc_info=exception)
