"""
Model Lifecycle Management System

This module handles:
- Model versioning
- Metadata storage
- Automatic rollback on performance degradation
- Model history tracking

Critical for production ML systems where model reliability and reproducibility
are essential (e.g., CERN-scale deployments).
"""

import json
import os
import pickle
from datetime import datetime
from config import MODEL_DIR, METADATA_PATH, CURRENT_MODEL_PATH, MIN_IMPROVEMENT_THRESHOLD, ROLLBACK_ON_DEGRADATION
from utils.logger import get_logger, log_model_switch


class ModelLifecycleManager:
    """
    Manages the complete lifecycle of ML models including versioning,
    metadata tracking, and automatic rollback.
    """
    
    def __init__(self):
        self.metadata_path = METADATA_PATH
        self.model_dir = MODEL_DIR
        self.current_model_path = CURRENT_MODEL_PATH
        self.logger = get_logger()
        self._ensure_metadata_exists()
    
    def _ensure_metadata_exists(self):
        """Initialize metadata file if it doesn't exist."""
        if not os.path.exists(self.metadata_path):
            metadata = {
                "current_version": 0,
                "models": {},
                "history": []
            }
            self._save_metadata(metadata)
    
    def _load_metadata(self):
        """Load model metadata from disk."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"current_version": 0, "models": {}, "history": []}
    
    def _save_metadata(self, metadata):
        """Save model metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_current_version(self):
        """Get the current model version number."""
        metadata = self._load_metadata()
        return metadata.get("current_version", 0)
    
    def get_current_accuracy(self):
        """Get the accuracy of the current model."""
        metadata = self._load_metadata()
        current_version = metadata.get("current_version", 0)
        if current_version > 0:
            model_info = metadata.get("models", {}).get(f"v{current_version}", {})
            return model_info.get("accuracy", None)
        return None
    
    def save_new_model(self, model, model_name, accuracy, dataset_size, drift_score=None, validation_metrics=None):
        """
        Save a new model version with metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., "RandomForest")
            accuracy: Validation accuracy
            dataset_size: Size of training dataset
            drift_score: Drift score at training time (optional)
            validation_metrics: Additional validation metrics (optional)
            
        Returns:
            int: New model version number
        """
        metadata = self._load_metadata()
        current_version = metadata.get("current_version", 0)
        new_version = current_version + 1
        
        # Prepare model data
        model_data = {
            "model": model,
            "model_name": model_name,
            "accuracy": accuracy
        }
        
        # Save versioned model
        versioned_path = f"{self.model_dir}/model_v{new_version}.pkl"
        with open(versioned_path, "wb") as f:
            pickle.dump(model_data, f)
        
        # Update metadata
        model_info = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "dataset_size": int(dataset_size),
            "training_timestamp": datetime.now().isoformat(),
            "model_path": versioned_path
        }
        
        if drift_score is not None:
            model_info["drift_score_at_training"] = float(drift_score)
        
        if validation_metrics:
            model_info["validation_metrics"] = validation_metrics
        
        metadata["models"][f"v{new_version}"] = model_info
        metadata["current_version"] = new_version
        
        # Add to history
        history_entry = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "action": "deploy",
            "model_name": model_name,
            "accuracy": float(accuracy)
        }
        metadata["history"].append(history_entry)
        
        # Keep only last 100 history entries
        if len(metadata["history"]) > 100:
            metadata["history"] = metadata["history"][-100:]
        
        self._save_metadata(metadata)
        
        # Update current model pointer
        self._update_current_model(model_data)
        
        self.logger.info(f"Saved new model version v{new_version} with accuracy {accuracy:.4f}")
        
        return new_version
    
    def _update_current_model(self, model_data):
        """Update the current_model.pkl file."""
        with open(self.current_model_path, "wb") as f:
            pickle.dump(model_data, f)
    
    def should_deploy_new_model(self, new_accuracy):
        """
        Determine if a new model should be deployed based on performance.
        
        Args:
            new_accuracy: Accuracy of the new model
            
        Returns:
            tuple: (should_deploy: bool, reason: str)
        """
        current_accuracy = self.get_current_accuracy()
        
        if current_accuracy is None:
            # No existing model, deploy the first one
            return True, "first_model"
        
        improvement = new_accuracy - current_accuracy
        
        if improvement >= MIN_IMPROVEMENT_THRESHOLD:
            return True, "performance_improvement"
        elif improvement < -MIN_IMPROVEMENT_THRESHOLD and ROLLBACK_ON_DEGRADATION:
            # New model is worse, but we'll deploy and rollback if needed
            return True, "deploy_with_rollback_check"
        elif abs(improvement) < MIN_IMPROVEMENT_THRESHOLD:
            return False, "no_significant_improvement"
        else:
            return True, "deploy_with_rollback_check"
    
    def deploy_model(self, new_version, new_accuracy):
        """
        Deploy a new model version, with automatic rollback if needed.
        
        Args:
            new_version: Version number to deploy
            new_accuracy: Accuracy of the new model
            
        Returns:
            bool: True if deployment succeeded, False if rolled back
        """
        metadata = self._load_metadata()
        current_version = metadata.get("current_version", 0)
        current_accuracy = self.get_current_accuracy()
        
        if new_version <= current_version:
            self.logger.warning(f"Attempted to deploy version {new_version} which is not newer than current {current_version}")
            return False
        
        # Check if we should rollback
        if current_accuracy is not None:
            improvement = new_accuracy - current_accuracy
            
            if improvement < -MIN_IMPROVEMENT_THRESHOLD and ROLLBACK_ON_DEGRADATION:
                # Performance degraded, rollback
                self.logger.warning(
                    f"New model v{new_version} accuracy {new_accuracy:.4f} is worse than "
                    f"current v{current_version} accuracy {current_accuracy:.4f}. Rolling back."
                )
                log_model_switch(new_version, current_version, "performance_degradation", new_accuracy, current_accuracy)
                return False
        
        # Deploy the new model
        model_info = metadata["models"][f"v{new_version}"]
        model_path = model_info["model_path"]
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self._update_current_model(model_data)
        metadata["current_version"] = new_version
        
        # Update history
        history_entry = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "action": "deploy",
            "model_name": model_info["model_name"],
            "accuracy": float(new_accuracy)
        }
        metadata["history"].append(history_entry)
        
        self._save_metadata(metadata)
        
        log_model_switch(current_version, new_version, "deployment", current_accuracy, new_accuracy)
        self.logger.info(f"Deployed model version v{new_version}")
        
        return True
    
    def rollback_to_version(self, target_version):
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Version number to rollback to
            
        Returns:
            bool: True if rollback succeeded
        """
        metadata = self._load_metadata()
        current_version = metadata.get("current_version", 0)
        
        if target_version >= current_version:
            self.logger.warning(f"Cannot rollback to version {target_version} (current: {current_version})")
            return False
        
        if f"v{target_version}" not in metadata["models"]:
            self.logger.error(f"Model version v{target_version} not found in metadata")
            return False
        
        # Load the target model
        model_info = metadata["models"][f"v{target_version}"]
        model_path = model_info["model_path"]
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Update current model
        self._update_current_model(model_data)
        metadata["current_version"] = target_version
        
        # Update history
        history_entry = {
            "version": target_version,
            "timestamp": datetime.now().isoformat(),
            "action": "rollback",
            "from_version": current_version,
            "model_name": model_info["model_name"],
            "accuracy": model_info["accuracy"]
        }
        metadata["history"].append(history_entry)
        
        self._save_metadata(metadata)
        
        log_model_switch(current_version, target_version, "rollback")
        self.logger.info(f"Rolled back to model version v{target_version}")
        
        return True
    
    def get_model_info(self, version=None):
        """
        Get information about a model version.
        
        Args:
            version: Version number (None for current version)
            
        Returns:
            dict: Model information
        """
        metadata = self._load_metadata()
        if version is None:
            version = metadata.get("current_version", 0)
        
        return metadata.get("models", {}).get(f"v{version}", {})
    
    def get_all_versions(self):
        """Get list of all model versions."""
        metadata = self._load_metadata()
        return list(metadata.get("models", {}).keys())
