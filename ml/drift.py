"""
Drift Detection Module for Adaptive Sentiment Analysis

This module implements data and concept drift detection using multiple methods:
1. Feature Distribution Drift: Measures changes in TF-IDF feature distributions
2. Confidence Drift: Measures changes in prediction confidence/entropy

Research Context:
- Drift detection is critical for production ML systems where data distributions
  change over time (e.g., social media sentiment shifts, domain adaptation)
- CERN-scale systems require automated drift detection to maintain model reliability
  across long-term deployments and changing data sources
"""

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from config import VECTORIZER_PATH, FEATURE_DRIFT_WEIGHT, CONFIDENCE_DRIFT_WEIGHT, DRIFT_THRESHOLD


def compute_feature_distribution_drift(training_features, incoming_features):
    """
    Compute feature distribution drift using Population Stability Index (PSI)
    and mean difference metrics.
    
    Args:
        training_features: Sparse matrix of training TF-IDF features
        incoming_features: Sparse matrix of incoming data TF-IDF features
        
    Returns:
        float: Feature drift score (0-1, higher = more drift)
    """
    # Convert sparse matrices to dense for statistical analysis
    # Use mean of features as summary statistic
    train_mean = np.array(training_features.mean(axis=0)).flatten()
    incoming_mean = np.array(incoming_features.mean(axis=0)).flatten()
    
    # Method 1: Mean absolute difference (normalized)
    mean_diff = np.mean(np.abs(train_mean - incoming_mean))
    # Normalize by training mean to get relative change
    normalized_diff = mean_diff / (np.mean(np.abs(train_mean)) + 1e-10)
    
    # Method 2: Population Stability Index (PSI) approximation
    # PSI measures distribution shift between two populations
    # We use KL divergence as a proxy for PSI
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    train_norm = np.abs(train_mean) + epsilon
    incoming_norm = np.abs(incoming_mean) + epsilon
    
    # Normalize to probability distributions
    train_prob = train_norm / train_norm.sum()
    incoming_prob = incoming_norm / incoming_norm.sum()
    
    # Compute KL divergence (asymmetric measure of distribution difference)
    kl_div = np.sum(train_prob * np.log(train_prob / (incoming_prob + epsilon) + epsilon))
    
    # Convert KL divergence to 0-1 scale (using sigmoid-like transformation)
    psi_score = 1 - np.exp(-kl_div)
    
    # Combine both metrics
    feature_drift = (normalized_diff + psi_score) / 2
    
    # Clamp to [0, 1]
    return min(1.0, max(0.0, feature_drift))


def compute_confidence_drift(training_predictions, incoming_predictions):
    """
    Compute prediction confidence drift using entropy and variance metrics.
    
    Confidence drift occurs when the model's prediction certainty changes,
    indicating potential concept drift (the relationship between features
    and labels has changed).
    
    Args:
        training_predictions: Array of prediction probabilities from training data
        incoming_predictions: Array of prediction probabilities from incoming data
        
    Returns:
        float: Confidence drift score (0-1, higher = more drift)
    """
    if len(training_predictions) == 0 or len(incoming_predictions) == 0:
        return 0.0
    
    # Method 1: Entropy-based drift
    # Higher entropy = lower confidence, lower entropy = higher confidence
    def compute_entropy(probs):
        """Compute average entropy of predictions."""
        # Handle both 1D (binary) and 2D (multi-class) probability arrays
        if probs.ndim == 1:
            # Binary classification: convert to [prob_neg, prob_pos]
            probs_2d = np.column_stack([1 - probs, probs])
        else:
            probs_2d = probs
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        probs_2d = np.clip(probs_2d, epsilon, 1 - epsilon)
        
        # Compute entropy: -sum(p * log(p))
        entropies = -np.sum(probs_2d * np.log(probs_2d), axis=1)
        return np.mean(entropies)
    
    train_entropy = compute_entropy(training_predictions)
    incoming_entropy = compute_entropy(incoming_predictions)
    
    # Entropy difference (normalized)
    entropy_diff = abs(train_entropy - incoming_entropy) / (train_entropy + 1e-10)
    
    # Method 2: Variance in confidence
    # Measure how spread out the confidence values are
    def compute_confidence_variance(probs):
        """Compute variance of maximum probabilities (confidence)."""
        if probs.ndim == 1:
            # For binary, use absolute distance from 0.5
            confidences = np.abs(probs - 0.5) * 2
        else:
            confidences = np.max(probs, axis=1)
        return np.var(confidences)
    
    train_var = compute_confidence_variance(training_predictions)
    incoming_var = compute_confidence_variance(incoming_predictions)
    
    # Variance difference (normalized)
    var_diff = abs(train_var - incoming_var) / (train_var + 1e-10) if train_var > 0 else 0.0
    
    # Combine entropy and variance metrics
    confidence_drift = (entropy_diff + var_diff) / 2
    
    # Clamp to [0, 1]
    return min(1.0, max(0.0, confidence_drift))


def detect_drift(training_data, incoming_data, model, vectorizer=None):
    """
    Main drift detection function that combines feature and confidence drift.
    
    This function implements a two-stage drift detection:
    1. Feature drift: Detects changes in input data distribution
    2. Confidence drift: Detects changes in model prediction patterns
    
    Research Note:
    - Feature drift alone may miss concept drift (when P(Y|X) changes but P(X) doesn't)
    - Confidence drift helps catch concept drift by monitoring prediction uncertainty
    - Combining both provides comprehensive drift detection
    
    Args:
        training_data: Training text data (list/array of strings)
        incoming_data: Incoming text data (list/array of strings)
        model: Trained sklearn model
        vectorizer: Fitted TF-IDF vectorizer (if None, loads from disk)
        
    Returns:
        dict: Drift detection results with scores and overall assessment
    """
    if vectorizer is None:
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
        else:
            raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")
    
    # Transform text to features
    training_features = vectorizer.transform(training_data)
    incoming_features = vectorizer.transform(incoming_data)
    
    # Compute feature distribution drift
    feature_drift = compute_feature_distribution_drift(
        training_features, incoming_features
    )
    
    # Get prediction probabilities for confidence drift
    # Handle both sparse and dense feature matrices
    if hasattr(training_features, 'toarray'):
        train_probs = model.predict_proba(training_features.toarray())
        incoming_probs = model.predict_proba(incoming_features.toarray())
    else:
        train_probs = model.predict_proba(training_features)
        incoming_probs = model.predict_proba(incoming_features)
    
    # For binary classification, extract positive class probability
    if train_probs.shape[1] == 2:
        train_probs = train_probs[:, 1]
        incoming_probs = incoming_probs[:, 1]
    
    # Compute confidence drift
    confidence_drift = compute_confidence_drift(train_probs, incoming_probs)
    
    # Weighted combination of drift scores
    overall_drift = (
        FEATURE_DRIFT_WEIGHT * feature_drift +
        CONFIDENCE_DRIFT_WEIGHT * confidence_drift
    )
    
    return {
        "feature_drift": float(feature_drift),
        "confidence_drift": float(confidence_drift),
        "overall_drift_score": float(overall_drift),
        "drift_detected": overall_drift > DRIFT_THRESHOLD
    }


def get_recent_training_data(limit=1000):
    """
    Retrieve recent training data for drift comparison.
    
    This function loads a sample of recent training data from the database
    to use as a baseline for drift detection.
    
    Args:
        limit: Maximum number of samples to retrieve
        
    Returns:
        pandas.DataFrame: Recent training data
    """
    from ml.data_loader import load_database
    
    data = load_database()
    if len(data) > limit:
        # Get most recent samples
        data = data.tail(limit)
    
    return data
