# Quick Start Guide

Get up and running with the Self-Training Sentiment Analysis System in 5 minutes!

## ⚡ Fast Setup

### 1. Prerequisites Check

```bash
# Check Python version (3.8+ required)
python --version

# Check MySQL (optional for basic testing)
mysql --version
```

### 2. Clone and Install

```bash
# Clone repository
git clone https://github.com/your-username/Setiment_analysis.git
cd Setiment_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Configuration

Edit `config.py`:

```python
# Minimal configuration for testing
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "sentiment_analysis"
}

# For testing without database, you can skip database setup initially
```

### 4. Run the Application

```bash
python app.py
```

Open browser: `http://localhost:5000`

## 🧪 Test Without Database

If you don't have MySQL set up yet, you can test the system:

1. **Skip database setup** (for now)
2. **Load a pre-trained model** or create a simple test model
3. **Test predictions** via the web interface

## 📊 Basic Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Enter text in the textarea
3. Click "Analyze Sentiment"
4. View prediction and confidence

### Python API

```python
from ml.trainer import train_pipeline

# Train a model
model_name, accuracy, drift_info = train_pipeline(force_retrain=True)
print(f"Trained {model_name} with accuracy: {accuracy:.4f}")
```

### Drift Detection

```python
from ml.drift import detect_drift
import pickle

# Load model
with open("models_store/current_model.pkl", "rb") as f:
    model = pickle.load(f)["model"]

# Detect drift
drift_info = detect_drift(
    training_data=["old text"],
    incoming_data=["new text"],
    model=model
)
print(f"Drift score: {drift_info['overall_drift_score']}")
```

## 🗄️ Database Setup (Optional)

### Create Database

```sql
CREATE DATABASE sentiment_analysis;
USE sentiment_analysis;

CREATE TABLE twitter_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    clean_text TEXT,
    category INT
);

CREATE TABLE data_flag (
    id INT PRIMARY KEY,
    last_row_count INT DEFAULT 0
);

INSERT INTO data_flag (id, last_row_count) VALUES (1, 0);
```

### Add Sample Data

```sql
INSERT INTO twitter_data (clean_text, category) VALUES
('I love this product!', 1),
('This is terrible', 0),
('Amazing experience', 1),
('Not worth the money', 0);
```

## 🚀 Next Steps

1. **Train Initial Model**:
   ```bash
   python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"
   ```

2. **Start Scheduler** (optional):
   ```bash
   python scheduler.py
   ```

3. **Explore Features**:
   - Enable self-learning in the web interface
   - Check system logs: `logs/system.log`
   - View model metadata: `models_store/metadata.json`

4. **Read Full Documentation**:
   - [README.md](README.md) - Complete overview
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [API.md](API.md) - API reference
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Production setup

## 🐛 Troubleshooting

### Import Errors

```bash
# Ensure you're in the project directory
cd Setiment_analysis

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Database Connection Error

- Check MySQL is running: `sudo systemctl status mysql`
- Verify credentials in `config.py`
- Test connection: `mysql -u root -p`

### Model Not Found

```bash
# Train initial model
python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"
```

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill process or change port in app.py
```

## 📖 Common Tasks

### View System Information

```python
from utils.model_lifecycle import ModelLifecycleManager

manager = ModelLifecycleManager()
print(f"Current version: {manager.get_current_version()}")
print(f"Model info: {manager.get_model_info()}")
```

### Check Logs

```bash
# View recent logs
tail -f logs/system.log

# Search for errors
grep ERROR logs/system.log
```

### Enable Experiment Mode

Edit `config.py`:
```python
EXPERIMENT_MODE = True
```

Metrics will be saved to `experiments/` directory.

## 🎓 Learning Resources

- **Drift Detection**: See `ml/drift.py` for implementation details
- **Model Lifecycle**: See `utils/model_lifecycle.py` for versioning
- **Training Pipeline**: See `ml/trainer.py` for training logic

## 💡 Tips

1. **Start Simple**: Begin with basic predictions, then enable advanced features
2. **Monitor Logs**: Check `logs/system.log` for system events
3. **Experiment Mode**: Enable for research and analysis
4. **Model Versions**: Check `models_store/metadata.json` for model history

## 🆘 Need Help?

- Check [README.md](README.md) for detailed information
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system understanding
- Open an issue on GitHub
- Check existing issues for solutions

---

**Ready to dive deeper?** Check out the [full documentation](README.md)!
