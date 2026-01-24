"""
Enhanced Training Pipeline with Drift Detection and Self-Learning

This module implements the complete training pipeline with:
- Drift detection before retraining
- Confidence-filtered self-learning
- Model lifecycle management
- Experiment tracking

Research Context:
- Adaptive retraining only when drift is detected reduces computational costs
- Confidence filtering prevents confirmation bias in self-learning
- Model versioning ensures reproducibility and enables rollback
"""

import time
from sklearn.model_selection import train_test_split
from ml.data_loader import load_database
from ml.preprocessing import create_vectorizer
from ml.models import MODELS
from ml.evaluator import evaluate
from ml.drift import detect_drift, get_recent_training_data
from config import (
    VECTORIZER_PATH, DRIFT_THRESHOLD, CONFIDENCE_THRESHOLD,
    EXPERIMENT_MODE, LOW_CONFIDENCE_DIR
)
from utils.logger import (
    get_logger, log_training_start, log_training_end,
    log_drift_detection, log_error
)
from utils.model_lifecycle import ModelLifecycleManager
from utils.experiment_tracker import ExperimentTracker
import pickle
import os
import pandas as pd


def train_pipeline(force_retrain=False, check_drift=True):
    """
    Main training pipeline with drift detection and self-learning.
    
    Args:
        force_retrain: If True, skip drift detection and retrain anyway
        check_drift: If True, check for drift before retraining
        
    Returns:
        tuple: (model_name, accuracy, drift_info)
    """
    logger = get_logger()
    lifecycle_manager = ModelLifecycleManager()
    experiment_tracker = ExperimentTracker() if EXPERIMENT_MODE else None
    
    start_time = time.time()
    
    try:
        # Load data
        data = load_database()
        data = data.dropna(subset=['category'])
        dataset_size = len(data)
        
        log_training_start(dataset_size=dataset_size)
        
        # Drift detection (if enabled and not forcing retrain)
        drift_info = None
        should_retrain = force_retrain
        
        if check_drift and not force_retrain:
            try:
                # Get baseline training data for comparison
                baseline_data = get_recent_training_data(limit=min(1000, len(data)))
                
                if len(baseline_data) > 0 and len(data) > 0:
                    # Load current model for drift detection
                    from config import CURRENT_MODEL_PATH
                    if os.path.exists(CURRENT_MODEL_PATH):
                        with open(CURRENT_MODEL_PATH, "rb") as f:
                            current_model_data = pickle.load(f)
                        current_model = current_model_data["model"]
                        
                        # Load vectorizer
                        with open(VECTORIZER_PATH, "rb") as f:
                            vectorizer = pickle.load(f)
                        
                        # Detect drift
                        drift_info = detect_drift(
                            training_data=baseline_data['clean_text'].tolist(),
                            incoming_data=data['clean_text'].tolist(),
                            model=current_model,
                            vectorizer=vectorizer
                        )
                        
                        should_retrain = drift_info.get("drift_detected", False)
                        
                        log_drift_detection(
                            drift_info["feature_drift"],
                            drift_info["confidence_drift"],
                            drift_info["overall_drift_score"],
                            triggered_retrain=should_retrain
                        )
                        
                        # Log to experiment tracker
                        if experiment_tracker:
                            experiment_tracker.log_drift(drift_info)
                    else:
                        # No existing model, must train
                        should_retrain = True
                        logger.info("No existing model found, training new model")
                else:
                    should_retrain = True
            except Exception as e:
                log_error("DRIFT_DETECTION_ERROR", str(e), e)
                # On error, proceed with training
                should_retrain = True
        
        if not should_retrain:
            logger.info(f"Drift score {drift_info['overall_drift_score']:.4f} below threshold {DRIFT_THRESHOLD}. Skipping retraining.")
            return None, None, drift_info
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            data.clean_text, data.category, test_size=0.2, stratify=data.category, random_state=42
        )
        
        # Create vectorizer
        vectorizer, X_train, X_test = create_vectorizer(X_train_text, X_test_text)
        
        # Evaluate models
        best_name, best_model, best_acc = evaluate(MODELS, X_train, y_train, X_test, y_test)
        
        # Save vectorizer
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        
        # Check if we should deploy this model
        should_deploy, deploy_reason = lifecycle_manager.should_deploy_new_model(best_acc)
        
        if should_deploy:
            # Save new model version with metadata
            drift_score = drift_info["overall_drift_score"] if drift_info else None
            new_version = lifecycle_manager.save_new_model(
                model=best_model,
                model_name=best_name,
                accuracy=best_acc,
                dataset_size=dataset_size,
                drift_score=drift_score
            )
            
            # Deploy the model
            deployment_success = lifecycle_manager.deploy_model(new_version, best_acc)
            
            if not deployment_success:
                logger.warning("Model deployment failed, may have been rolled back")
        else:
            logger.info(f"New model accuracy {best_acc:.4f} does not meet deployment criteria: {deploy_reason}")
        
        training_time = time.time() - start_time
        log_training_end(best_name, best_acc, training_time)
        
        # Log to experiment tracker
        if experiment_tracker:
            experiment_tracker.log_training(
                model_name=best_name,
                accuracy=best_acc,
                dataset_size=dataset_size,
                training_time=training_time,
                drift_score=drift_info["overall_drift_score"] if drift_info else None
            )
        
        return best_name, best_acc, drift_info
        
    except Exception as e:
        log_error("TRAINING_ERROR", str(e), e)
        raise


def add_self_learning_sample(text, prediction, confidence, label=None):
    """
    Add a high-confidence prediction to the training database for self-learning.
    
    This function implements confidence-filtered self-learning:
    - Only high-confidence predictions are accepted (prevents confirmation bias)
    - Low-confidence samples are stored separately for manual review
    - This ensures the model learns from reliable predictions only
    
    Research Note:
    Confirmation bias occurs when a model reinforces its own mistakes.
    By only accepting high-confidence predictions (confidence > threshold),
    we ensure the model only learns from cases where it's highly certain,
    reducing the risk of propagating errors.
    
    Args:
        text: Input text
        prediction: Predicted label (0 or 1)
        confidence: Prediction confidence/probability (0-1)
        label: Actual label if available (optional, for validation)
        
    Returns:
        bool: True if sample was accepted, False if rejected
    """
    logger = get_logger()
    
    if confidence >= CONFIDENCE_THRESHOLD:
        # High confidence: add to training database
        try:
            from database.db import get_connection
            
            conn = get_connection()
            cursor = conn.cursor()
            
            # Insert into database (assuming table structure)
            # Adjust SQL based on your actual schema
            cursor.execute(
                "INSERT INTO twitter_data (clean_text, category) VALUES (%s, %s)",
                (text, int(prediction))
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Added high-confidence sample to training data (confidence={confidence:.4f})")
            return True
            
        except Exception as e:
            log_error("SELF_LEARNING_ERROR", f"Failed to add sample: {str(e)}", e)
            return False
    else:
        # Low confidence: store separately for review
        try:
            os.makedirs(LOW_CONFIDENCE_DIR, exist_ok=True)
            low_conf_file = os.path.join(LOW_CONFIDENCE_DIR, "low_confidence_samples.txt")
            
            with open(low_conf_file, "a", encoding="utf-8") as f:
                f.write(f"{text}\t{prediction}\t{confidence:.4f}\t{label if label else 'unknown'}\n")
            
            logger.debug(f"Stored low-confidence sample (confidence={confidence:.4f})")
            return False
            
        except Exception as e:
            log_error("LOW_CONFIDENCE_STORAGE_ERROR", str(e), e)
            return False
