"""
Experiment Tracking Module for Research Mode

This module tracks metrics, drift scores, and model performance over time
for research and analysis purposes. Enables the system to function as
a research platform rather than just an application.
"""

import json
import os
from datetime import datetime
from config import EXPERIMENTS_DIR, METRICS_LOG_PATH, DRIFT_LOG_PATH
from utils.logger import get_logger


class ExperimentTracker:
    """
    Tracks experiments, metrics, and drift scores for research analysis.
    """
    
    def __init__(self):
        self.metrics_path = METRICS_LOG_PATH
        self.drift_path = DRIFT_LOG_PATH
        self.logger = get_logger()
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Initialize experiment tracking files if they don't exist."""
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.drift_path):
            with open(self.drift_path, 'w') as f:
                json.dump([], f)
    
    def log_training(self, model_name, accuracy, dataset_size, training_time, drift_score=None):
        """
        Log training metrics.
        
        Args:
            model_name: Name of trained model
            accuracy: Validation accuracy
            dataset_size: Size of training dataset
            training_time: Time taken for training (seconds)
            drift_score: Drift score that triggered training (optional)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "accuracy": float(accuracy),
            "dataset_size": int(dataset_size),
            "training_time": float(training_time),
            "drift_score": float(drift_score) if drift_score is not None else None
        }
        
        try:
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
            
            metrics.append(entry)
            
            # Keep only last 1000 entries
            if len(metrics) > 1000:
                metrics = metrics[-1000:]
            
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.debug(f"Logged training metrics: {model_name} accuracy={accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to log training metrics: {e}")
    
    def log_drift(self, drift_info):
        """
        Log drift detection event.
        
        Args:
            drift_info: Dictionary with drift scores from detect_drift()
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "feature_drift": float(drift_info.get("feature_drift", 0)),
            "confidence_drift": float(drift_info.get("confidence_drift", 0)),
            "overall_drift_score": float(drift_info.get("overall_drift_score", 0)),
            "drift_detected": bool(drift_info.get("drift_detected", False))
        }
        
        try:
            with open(self.drift_path, 'r') as f:
                drift_logs = json.load(f)
            
            drift_logs.append(entry)
            
            # Keep only last 1000 entries
            if len(drift_logs) > 1000:
                drift_logs = drift_logs[-1000:]
            
            with open(self.drift_path, 'w') as f:
                json.dump(drift_logs, f, indent=2)
            
            self.logger.debug(f"Logged drift detection: overall={entry['overall_drift_score']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to log drift: {e}")
    
    def get_training_history(self, limit=None):
        """
        Get training history.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            list: Training history entries
        """
        try:
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
            
            if limit:
                return metrics[-limit:]
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to load training history: {e}")
            return []
    
    def get_drift_history(self, limit=None):
        """
        Get drift detection history.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            list: Drift history entries
        """
        try:
            with open(self.drift_path, 'r') as f:
                drift_logs = json.load(f)
            
            if limit:
                return drift_logs[-limit:]
            return drift_logs
            
        except Exception as e:
            self.logger.error(f"Failed to load drift history: {e}")
            return []
