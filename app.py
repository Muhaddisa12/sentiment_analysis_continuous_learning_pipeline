"""
Flask Application for Self-Training Sentiment Analysis System

This application provides a web interface for:
- Real-time sentiment prediction
- Model information display (version, drift, confidence)
- Self-learning integration with confidence filtering
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from config import CURRENT_MODEL_PATH, VECTORIZER_PATH, CONFIDENCE_THRESHOLD
from utils.model_lifecycle import ModelLifecycleManager
from utils.logger import get_logger
from ml.trainer import add_self_learning_sample

app = Flask(__name__)
logger = get_logger()
lifecycle_manager = ModelLifecycleManager()


def load_model_and_vectorizer():
    """Load current model and vectorizer from disk."""
    try:
        with open(CURRENT_MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        
        return model_data["model"], model_data["model_name"], vectorizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.route('/')
def home():
    """Main page with system information."""
    if os.path.exists("maintenance.flag"):
        return render_template("maintenance.html")
    
    # Get system information
    try:
        current_version = lifecycle_manager.get_current_version()
        model_info = lifecycle_manager.get_model_info()
        
        system_info = {
            "model_version": current_version,
            "model_name": model_info.get("model_name", "Unknown"),
            "accuracy": model_info.get("accuracy"),
            "last_training": model_info.get("training_timestamp"),
            "drift_score": model_info.get("drift_score_at_training")
        }
    except Exception as e:
        logger.error(f"Failed to load system info: {e}")
        system_info = {
            "model_version": 0,
            "model_name": "Unknown",
            "accuracy": None,
            "last_training": None,
            "drift_score": None
        }
    
    return render_template("index.html", system_info=system_info)


@app.route('/predict', methods=["POST"])
def predict():
    """Predict sentiment and optionally add to self-learning."""
    try:
        model, model_name, vectorizer = load_model_and_vectorizer()
        
        text = request.form.get("text_input", "").strip()
        if not text:
            return render_template("index.html", error="Please enter some text")
        
        # Transform text
        X = vectorizer.transform([text])
        
        # Get prediction and probabilities
        pred = model.predict(X)[0]
        
        # Get prediction probabilities for confidence calculation
        if hasattr(model, 'predict_proba'):
            if hasattr(X, 'toarray'):
                probs = model.predict_proba(X.toarray())[0]
            else:
                probs = model.predict_proba(X)[0]
            
            # For binary classification, get confidence as max probability
            confidence = float(np.max(probs))
            # For binary, also get the positive class probability
            if len(probs) == 2:
                positive_prob = float(probs[1])
            else:
                positive_prob = confidence
        else:
            # Model doesn't support probabilities, use default confidence
            confidence = 0.5
            positive_prob = 0.5
        
        sentiment = "Positive Sentiment" if pred == 1 else "Negative Sentiment"
        
        # Self-learning: add high-confidence predictions to training data
        enable_self_learning = request.form.get("enable_self_learning", "false").lower() == "true"
        if enable_self_learning:
            add_self_learning_sample(text, pred, confidence)
        
        # Get system information
        current_version = lifecycle_manager.get_current_version()
        model_info = lifecycle_manager.get_model_info()
        
        system_info = {
            "model_version": current_version,
            "model_name": model_name,
            "accuracy": model_info.get("accuracy"),
            "last_training": model_info.get("training_timestamp"),
            "drift_score": model_info.get("drift_score_at_training")
        }
        
        return render_template(
            "index.html",
            prediction=sentiment,
            model=model_name,
            input_text=text,
            confidence=confidence,
            positive_prob=positive_prob,
            system_info=system_info,
            self_learning_enabled=enable_self_learning,
            high_confidence=confidence >= CONFIDENCE_THRESHOLD
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error=f"Prediction failed: {str(e)}")


@app.route('/api/system_info', methods=["GET"])
def get_system_info():
    """API endpoint for system information."""
    try:
        current_version = lifecycle_manager.get_current_version()
        model_info = lifecycle_manager.get_model_info()
        
        return jsonify({
            "model_version": current_version,
            "model_name": model_info.get("model_name", "Unknown"),
            "accuracy": model_info.get("accuracy"),
            "last_training": model_info.get("training_timestamp"),
            "drift_score": model_info.get("drift_score_at_training"),
            "dataset_size": model_info.get("dataset_size")
        })
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
