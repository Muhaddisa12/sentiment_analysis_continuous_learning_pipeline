# API Documentation

This document describes the API endpoints and usage for the Self-Training Sentiment Analysis System.

## 🌐 Base URL

```
http://localhost:5000
```

## 📋 Endpoints

### 1. Home Page

**Endpoint**: `GET /`

**Description**: Main web interface for sentiment analysis.

**Response**: HTML page with:
- System information (model version, accuracy, drift score)
- Prediction form
- Results display

**Example**:
```bash
curl http://localhost:5000/
```

---

### 2. Predict Sentiment

**Endpoint**: `POST /predict`

**Description**: Predicts sentiment for input text and optionally adds to self-learning.

**Request Format**: `application/x-www-form-urlencoded`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text_input` | string | Yes | Text to analyze |
| `enable_self_learning` | boolean | No | Enable self-learning (default: false) |

**Response**: HTML page with:
- Prediction result (Positive/Negative)
- Confidence score
- Model information
- Self-learning status

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict \
  -d "text_input=I love this product!" \
  -d "enable_self_learning=true"
```

**Example Response** (HTML):
```html
<div class="result-card">
  <h2>Positive Sentiment</h2>
  <div class="confidence-info">
    <span>Confidence: 92.5%</span>
  </div>
</div>
```

---

### 3. System Information API

**Endpoint**: `GET /api/system_info`

**Description**: Returns current system information in JSON format.

**Response Format**: `application/json`

**Response Schema**:
```json
{
  "model_version": 2,
  "model_name": "RandomForest",
  "accuracy": 0.85,
  "last_training": "2024-01-20T10:30:00",
  "drift_score": 0.25,
  "dataset_size": 10000
}
```

**Field Descriptions**:
| Field | Type | Description |
|-------|------|-------------|
| `model_version` | integer | Current model version number |
| `model_name` | string | Algorithm name (e.g., "RandomForest") |
| `accuracy` | float | Validation accuracy (0-1) |
| `last_training` | string | ISO 8601 timestamp of last training |
| `drift_score` | float | Drift score at last training (0-1) |
| `dataset_size` | integer | Size of training dataset |

**Example Request**:
```bash
curl http://localhost:5000/api/system_info
```

**Example Response**:
```json
{
  "model_version": 2,
  "model_name": "RandomForest",
  "accuracy": 0.85,
  "last_training": "2024-01-20T10:30:00",
  "drift_score": 0.25,
  "dataset_size": 10000
}
```

**Error Response** (500):
```json
{
  "error": "Failed to load model metadata"
}
```

---

## 🔧 Python API

### Training Pipeline

**Module**: `ml.trainer`

**Function**: `train_pipeline(force_retrain=False, check_drift=True)`

**Description**: Main training pipeline with drift detection.

**Parameters**:
- `force_retrain` (bool): Skip drift detection and retrain anyway
- `check_drift` (bool): Enable drift detection before retraining

**Returns**: `tuple(model_name, accuracy, drift_info)`

**Example**:
```python
from ml.trainer import train_pipeline

# Train with drift detection
model_name, accuracy, drift_info = train_pipeline(check_drift=True)

# Force retrain
model_name, accuracy, drift_info = train_pipeline(force_retrain=True)
```

---

### Drift Detection

**Module**: `ml.drift`

**Function**: `detect_drift(training_data, incoming_data, model, vectorizer=None)`

**Description**: Detects drift between training and incoming data.

**Parameters**:
- `training_data` (list/array): Baseline training text data
- `incoming_data` (list/array): New incoming text data
- `model`: Trained sklearn model
- `vectorizer` (optional): TF-IDF vectorizer (loads from disk if None)

**Returns**: `dict` with drift scores

**Example**:
```python
from ml.drift import detect_drift
import pickle

# Load model
with open("models_store/current_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]

# Detect drift
drift_info = detect_drift(
    training_data=["old text 1", "old text 2"],
    incoming_data=["new text 1", "new text 2"],
    model=model
)

print(f"Overall drift: {drift_info['overall_drift_score']}")
print(f"Drift detected: {drift_info['drift_detected']}")
```

**Response Schema**:
```python
{
    "feature_drift": 0.15,        # Feature distribution drift (0-1)
    "confidence_drift": 0.20,      # Confidence drift (0-1)
    "overall_drift_score": 0.17,  # Weighted combination (0-1)
    "drift_detected": False        # True if > threshold
}
```

---

### Model Lifecycle Management

**Module**: `utils.model_lifecycle`

**Class**: `ModelLifecycleManager`

**Methods**:

#### `get_current_version()`
Returns current model version number.

**Example**:
```python
from utils.model_lifecycle import ModelLifecycleManager

manager = ModelLifecycleManager()
version = manager.get_current_version()
print(f"Current version: {version}")
```

#### `save_new_model(model, model_name, accuracy, dataset_size, drift_score=None)`
Saves a new model version with metadata.

**Example**:
```python
version = manager.save_new_model(
    model=trained_model,
    model_name="RandomForest",
    accuracy=0.85,
    dataset_size=10000,
    drift_score=0.25
)
```

#### `deploy_model(new_version, new_accuracy)`
Deploys a model version (with automatic rollback if needed).

**Example**:
```python
success = manager.deploy_model(new_version=3, new_accuracy=0.87)
if success:
    print("Model deployed successfully")
else:
    print("Deployment failed or rolled back")
```

#### `rollback_to_version(target_version)`
Rolls back to a previous model version.

**Example**:
```python
success = manager.rollback_to_version(target_version=1)
```

#### `get_model_info(version=None)`
Gets information about a model version.

**Example**:
```python
info = manager.get_model_info(version=2)
print(f"Accuracy: {info['accuracy']}")
print(f"Training time: {info['training_timestamp']}")
```

---

### Self-Learning

**Module**: `ml.trainer`

**Function**: `add_self_learning_sample(text, prediction, confidence, label=None)`

**Description**: Adds a high-confidence prediction to training data.

**Parameters**:
- `text` (str): Input text
- `prediction` (int): Predicted label (0 or 1)
- `confidence` (float): Prediction confidence (0-1)
- `label` (int, optional): Actual label if available

**Returns**: `bool` - True if accepted, False if rejected

**Example**:
```python
from ml.trainer import add_self_learning_sample

accepted = add_self_learning_sample(
    text="This is great!",
    prediction=1,
    confidence=0.92
)

if accepted:
    print("Sample added to training data")
else:
    print("Sample rejected (low confidence)")
```

---

### Experiment Tracking

**Module**: `utils.experiment_tracker`

**Class**: `ExperimentTracker`

**Methods**:

#### `log_training(model_name, accuracy, dataset_size, training_time, drift_score=None)`
Logs training metrics.

**Example**:
```python
from utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
tracker.log_training(
    model_name="RandomForest",
    accuracy=0.85,
    dataset_size=10000,
    training_time=120.5,
    drift_score=0.25
)
```

#### `log_drift(drift_info)`
Logs drift detection event.

**Example**:
```python
tracker.log_drift({
    "feature_drift": 0.15,
    "confidence_drift": 0.20,
    "overall_drift_score": 0.17,
    "drift_detected": False
})
```

#### `get_training_history(limit=None)`
Gets training history.

**Example**:
```python
history = tracker.get_training_history(limit=10)
for entry in history:
    print(f"{entry['timestamp']}: {entry['accuracy']}")
```

#### `get_drift_history(limit=None)`
Gets drift detection history.

**Example**:
```python
drift_history = tracker.get_drift_history(limit=50)
```

---

## 🔐 Authentication

Currently, the API does not require authentication. For production deployments, consider:

1. API key authentication
2. JWT tokens
3. OAuth 2.0
4. Rate limiting

## ⚠️ Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": "Error message description"
}
```

## 📊 Rate Limiting

Currently no rate limiting is implemented. For production, consider:

- Per-IP rate limiting
- Per-user rate limiting
- Request throttling

## 🔄 Webhooks (Future)

Potential webhook events:
- Model training completed
- Drift detected
- Model deployed
- Model rolled back

## 📝 Examples

### Complete Prediction Workflow

```python
import requests

# Get system info
response = requests.get("http://localhost:5000/api/system_info")
system_info = response.json()
print(f"Current model: {system_info['model_name']} v{system_info['model_version']}")

# Make prediction
response = requests.post(
    "http://localhost:5000/predict",
    data={
        "text_input": "I love this product!",
        "enable_self_learning": "true"
    }
)
# Parse HTML response for results
```

### Python Training Workflow

```python
from ml.trainer import train_pipeline
from utils.model_lifecycle import ModelLifecycleManager
from utils.experiment_tracker import ExperimentTracker

# Initialize
manager = ModelLifecycleManager()
tracker = ExperimentTracker()

# Train with drift detection
model_name, accuracy, drift_info = train_pipeline(check_drift=True)

if drift_info and drift_info['drift_detected']:
    print(f"Drift detected! Retrained model: {model_name}")
    print(f"New accuracy: {accuracy:.4f}")
else:
    print("No significant drift, model unchanged")
```

## 🧪 Testing the API

### Using curl

```bash
# Get system info
curl http://localhost:5000/api/system_info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -d "text_input=This is amazing!" \
  -d "enable_self_learning=false"
```

### Using Python requests

```python
import requests

# System info
response = requests.get("http://localhost:5000/api/system_info")
print(response.json())

# Prediction
response = requests.post(
    "http://localhost:5000/predict",
    data={"text_input": "Great product!"}
)
print(response.text)  # HTML response
```

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [REST API Best Practices](https://restfulapi.net/)
- [HTTP Status Codes](https://httpstatuses.com/)
