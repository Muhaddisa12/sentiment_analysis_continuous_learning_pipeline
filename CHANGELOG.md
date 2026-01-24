# Changelog

All notable changes to the Self-Training Sentiment Analysis System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-20

### 🎉 Major Release: CERN-Worthy Upgrade

This release transforms the project into a research-grade, drift-aware MLOps pipeline suitable for large-scale scientific data systems.

### Added

#### Drift Detection
- **Feature Distribution Drift**: Measures changes in TF-IDF feature distributions using mean difference and PSI approximation
- **Confidence Drift**: Measures changes in prediction confidence using entropy and variance metrics
- **Combined Drift Score**: Weighted combination of feature and confidence drift
- **Drift-triggered Retraining**: Automatic retraining only when drift is detected

#### Model Lifecycle Management
- **Model Versioning**: Automatic versioning of models (v1, v2, etc.)
- **Metadata Storage**: Comprehensive metadata including accuracy, timestamp, dataset size, drift score
- **Automatic Rollback**: Rollback to previous version if new model performs worse
- **Model History**: Complete history of model deployments and rollbacks

#### Self-Learning Improvements
- **Confidence Filtering**: Only high-confidence predictions (threshold: 0.85) added to training data
- **Low-Confidence Storage**: Low-confidence samples stored separately for manual review
- **Confirmation Bias Prevention**: Research rationale documented in code

#### Observability
- **Structured Logging**: Human-readable, timestamped logs for all system events
- **Experiment Tracking**: Research mode for tracking metrics and drift scores over time
- **Log Persistence**: Persistent log files in `logs/` directory

#### Enhanced GUI
- **System Information Display**: Model version, accuracy, drift score
- **Confidence Visualization**: Visual confidence bar and indicators
- **Self-Learning Toggle**: User control over self-learning feature
- **Real-time Status**: Live system status information

#### Documentation
- **Comprehensive README**: Professional documentation with architecture, research motivation, CERN relevance
- **API Documentation**: Complete API reference
- **Architecture Documentation**: Detailed system architecture
- **Deployment Guide**: Production deployment instructions
- **Contributing Guidelines**: Contribution guidelines and code standards

### Changed

- **Package Structure**: Proper Python package structure with `__init__.py` files
- **Import System**: Fixed all imports to use absolute imports
- **Configuration**: Centralized configuration in `config.py` with comprehensive settings
- **Training Pipeline**: Enhanced with drift detection and model lifecycle integration
- **Data Loading**: Updated to use configuration instead of hardcoded credentials
- **Scheduler**: Improved with drift-aware retraining and better error handling

### Fixed

- **Import Errors**: Fixed all relative import issues
- **Hardcoded Paths**: Removed hardcoded paths, using configuration
- **Module Structure**: Proper package structure for all modules
- **Database Connection**: Improved error handling and configuration

### Technical Details

- **Drift Detection Algorithm**: Two-stage detection (feature + confidence)
- **Model Storage**: Versioned models with metadata in JSON
- **Experiment Tracking**: JSON-based metrics and drift history
- **Logging Format**: Structured logs with timestamps and levels

### Research Contributions

- Demonstrates adaptive ML principles
- Implements drift detection for production systems
- Shows confidence-filtered self-learning
- Provides MLOps best practices (versioning, rollback, observability)
- Enables research reproducibility through experiment tracking

---

## [1.0.0] - 2024-01-01

### Added

- Initial Flask web application
- Basic sentiment prediction
- MySQL database integration
- Model training pipeline
- Multiple ML model support (Naive Bayes, Logistic Regression, SVM, KNN, Decision Tree, Random Forest)
- TF-IDF vectorization
- Model persistence using pickle
- Basic GUI with modern design
- Automated scheduler for retraining

### Features

- Real-time sentiment prediction
- Continuous learning from new data
- Model evaluation and selection
- Basic error handling

---

## [Unreleased]

### Planned

- Real-time drift monitoring dashboard
- Advanced drift detection methods (KS test, MMD)
- A/B testing framework
- Automated hyperparameter tuning
- Distributed training support
- Model explainability (SHAP, LIME)
- Visualization dashboard for experiments
- REST API authentication
- Rate limiting
- Webhook support
- Docker Compose setup
- Kubernetes deployment configs
- Performance optimization
- Unit and integration tests
- CI/CD pipeline

---

## Version History

- **v2.0.0**: CERN-worthy upgrade with drift detection, model lifecycle, and research features
- **v1.0.0**: Initial release with basic sentiment analysis functionality

---

## Migration Guide

### From v1.0.0 to v2.0.0

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update Configuration**:
   - Review new settings in `config.py`
   - Set `EXPERIMENT_MODE = True` for research tracking
   - Configure drift thresholds

3. **Database Migration**:
   - No schema changes required
   - Existing models will be versioned on next training

4. **Model Migration**:
   - Existing `current_model.pkl` will be migrated to v1
   - Metadata will be created automatically

5. **Code Updates**:
   - Update imports if using modules directly
   - Review new API endpoints
   - Check logging configuration

---

## Breaking Changes

### v2.0.0

- **Import Paths**: Some internal imports changed (use package imports)
- **Model Storage**: Models now versioned (old single model file still works)
- **Configuration**: New configuration options required

---

## Deprecations

None in current version.

---

## Security

### v2.0.0

- Improved error handling (no sensitive data in errors)
- Configuration-based credentials (should use environment variables in production)
- Input validation for predictions

---

## Acknowledgments

- CERN OpenLab for research inspiration
- scikit-learn community
- Flask community
- All contributors

---

For detailed information about changes, see:
- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API.md](API.md)
