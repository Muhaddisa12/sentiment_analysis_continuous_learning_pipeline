# System Architecture

This document provides a detailed technical overview of the Self-Training Sentiment Analysis System architecture.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │  Flask Web App │  │  REST API      │  │  Scheduler     │        │
│  │  (app.py)      │  │  Endpoints     │  │  (scheduler.py)│        │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘        │
└───────────┼────────────────────┼────────────────────┼────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Business Logic Layer                         │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              ML Pipeline (ml/)                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │ Trainer  │  │  Drift   │  │ Evaluator │  │ Models   │ │   │
│  │  │          │◄─┤ Detection│  │           │  │           │ │   │
│  │  └────┬─────┘  └──────────┘  └────┬──────┘  └──────────┘ │   │
│  │       │                            │                      │   │
│  │       └──────────┬─────────────────┘                      │   │
│  │                  ▼                                        │   │
│  │         ┌────────────────┐                                │   │
│  │         │ Preprocessing   │                                │   │
│  │         │ Data Loader     │                                │   │
│  │         └────────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Utilities (utils/)                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │   Logger     │  │ Model        │  │ Experiment   │      │   │
│  │  │              │  │ Lifecycle    │  │ Tracker      │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   MySQL      │  │  File System │  │  JSON Files   │             │
│  │  Database    │  │  (Models)    │  │  (Metadata)   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└───────────────────────────────────────────────────────────────────────┘
```

## 📦 Module Structure

### 1. Presentation Layer

#### `app.py` - Flask Web Application
- **Purpose**: Web interface for sentiment prediction
- **Key Components**:
  - `/` - Main page with system information
  - `/predict` - Sentiment prediction endpoint
  - `/api/system_info` - System information API
- **Responsibilities**:
  - Handle user requests
  - Display predictions and confidence
  - Integrate self-learning
  - Show system status

#### `scheduler.py` - Automated Retraining
- **Purpose**: Monitor database and trigger retraining
- **Key Features**:
  - Checks for new data at midnight
  - Triggers drift-aware retraining
  - Logs all events

### 2. Machine Learning Module (`ml/`)

#### `trainer.py` - Training Pipeline
**Flow**:
```
1. Load data from database
2. Check for drift (if enabled)
3. If drift detected or forced:
   a. Split data (train/test)
   b. Create vectorizer
   c. Evaluate models
   d. Select best model
   e. Save with versioning
   f. Deploy if better
4. Return results
```

**Key Functions**:
- `train_pipeline(force_retrain, check_drift)` - Main training function
- `add_self_learning_sample()` - Confidence-filtered self-learning

#### `drift.py` - Drift Detection
**Algorithms**:

1. **Feature Distribution Drift**:
   ```
   - Compute mean TF-IDF vectors
   - Calculate normalized mean difference
   - Compute KL divergence (PSI approximation)
   - Combine: (normalized_diff + psi_score) / 2
   ```

2. **Confidence Drift**:
   ```
   - Compute prediction entropy
   - Calculate confidence variance
   - Combine: (entropy_diff + var_diff) / 2
   ```

3. **Overall Drift**:
   ```
   overall = (feature_weight * feature_drift) + 
             (confidence_weight * confidence_drift)
   ```

**Key Functions**:
- `detect_drift()` - Main drift detection
- `compute_feature_distribution_drift()` - Feature drift calculation
- `compute_confidence_drift()` - Confidence drift calculation

#### `models.py` - Model Definitions
- Contains sklearn model configurations
- Models tested: Naive Bayes, Logistic Regression, SVM, KNN, Decision Tree, Random Forest

#### `evaluator.py` - Model Evaluation
- Trains multiple models
- Evaluates on test set
- Returns best model based on accuracy

#### `preprocessing.py` - Text Preprocessing
- TF-IDF vectorization
- Feature extraction

#### `data_loader.py` - Data Loading
- Connects to MySQL database
- Loads training data
- Uses configuration from `config.py`

### 3. Utilities Module (`utils/`)

#### `logger.py` - Structured Logging
**Log Levels**:
- INFO: Training events, drift detection, model switches
- ERROR: System errors
- DEBUG: Detailed debugging (when enabled)

**Log Format**:
```
TIMESTAMP | LEVEL | MODULE | MESSAGE
```

**Key Functions**:
- `log_training_start/end()` - Training events
- `log_drift_detection()` - Drift events
- `log_model_switch()` - Model deployment/rollback
- `log_error()` - Error logging

#### `model_lifecycle.py` - Model Lifecycle Management
**Model Versioning**:
```
models_store/
├── model_v1.pkl
├── model_v2.pkl
├── current_model.pkl
└── metadata.json
```

**Metadata Structure**:
```json
{
  "current_version": 2,
  "models": {
    "v1": {
      "model_name": "RandomForest",
      "accuracy": 0.85,
      "training_timestamp": "2024-01-20T10:30:00",
      "dataset_size": 10000,
      "drift_score_at_training": 0.25
    }
  },
  "history": [...]
}
```

**Key Features**:
- Automatic versioning
- Performance-based deployment
- Automatic rollback on degradation
- History tracking

#### `experiment_tracker.py` - Experiment Tracking
**Tracks**:
- Training metrics over time
- Drift scores history
- Model performance evolution

**Storage**:
- `experiments/metrics_history.json`
- `experiments/drift_history.json`

### 4. Database Module (`database/`)

#### `db.py` - Database Connection
- MySQL connection management
- Uses configuration from `config.py`

## 🔄 Data Flow

### Prediction Flow
```
User Input → Flask App → Load Model/Vectorizer → 
Transform Text → Predict → Get Confidence → 
Display Result → (Optional) Self-Learning
```

### Training Flow
```
Scheduler/Manual Trigger → Load Data → 
Drift Detection → (If drift) → Split Data → 
Train Models → Evaluate → Select Best → 
Version & Save → Deploy if Better
```

### Drift Detection Flow
```
Load Baseline Data → Load Current Model → 
Transform Both Datasets → Compute Feature Drift → 
Get Predictions → Compute Confidence Drift → 
Combine Scores → Return Result
```

## 🔐 Configuration Management

All configuration in `config.py`:
- Database credentials
- Model paths
- Drift thresholds
- Self-learning thresholds
- Experiment mode
- Logging settings

## 📊 State Management

### Model State
- Stored in `models_store/metadata.json`
- Tracks current version
- Maintains history

### Experiment State
- Stored in `experiments/` directory
- JSON files for metrics and drift
- Append-only for history

### Log State
- Stored in `logs/system.log`
- Rotating logs (manual management)
- Structured format

## 🚀 Scalability Considerations

### Current Limitations
- Single-threaded Flask app
- In-memory model loading
- Sequential training

### Future Enhancements
- Multi-worker Flask (Gunicorn)
- Model caching/loading optimization
- Distributed training support
- Database connection pooling
- Redis for model caching

## 🔒 Security Considerations

1. **Database Credentials**: Stored in config (should use environment variables in production)
2. **Input Validation**: Text input sanitization
3. **Error Handling**: No sensitive data in error messages
4. **File Permissions**: Model files should be protected

## 🧪 Testing Strategy

### Unit Tests
- Individual functions
- Mock dependencies
- Edge cases

### Integration Tests
- Full pipeline
- Database interactions
- Model lifecycle

### System Tests
- End-to-end workflows
- Error scenarios
- Performance tests

## 📈 Monitoring & Observability

### Logging
- Structured logs
- Timestamped events
- Error tracking

### Metrics (Future)
- Prediction latency
- Model accuracy over time
- Drift score trends
- Resource usage

## 🔄 Deployment Architecture

### Development
```
Local Machine → Flask Dev Server → Local MySQL
```

### Production (Recommended)
```
Load Balancer → Gunicorn Workers → 
Application Servers → MySQL Database
```

### Containerized (Future)
```
Docker Compose:
  - Flask App Container
  - MySQL Container
  - Redis Container (optional)
  - Nginx (optional)
```

## 📚 Dependencies

### Core
- Flask: Web framework
- scikit-learn: ML models
- pandas: Data manipulation
- numpy: Numerical operations

### Database
- sqlalchemy: ORM
- mysql-connector-python: MySQL driver

### Utilities
- scipy: Statistical functions

## 🎯 Design Principles

1. **Modularity**: Clear separation of concerns
2. **Configurability**: All settings in config.py
3. **Observability**: Comprehensive logging
4. **Reproducibility**: Versioned models and experiments
5. **Maintainability**: Clean code, documentation
6. **Extensibility**: Easy to add new models/features

## 🔮 Future Architecture Enhancements

1. **Microservices**: Split into separate services
2. **Message Queue**: Async processing
3. **Model Registry**: Centralized model management
4. **Feature Store**: Reusable feature engineering
5. **A/B Testing**: Model comparison framework
6. **Monitoring Dashboard**: Real-time metrics
