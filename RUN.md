# How to Run the Application

##  Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd Setiment_analysis

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Configure Database (Optional for Basic Testing)

If you have MySQL set up, edit `config.py`:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",           # Your MySQL username
    "password": "",            # Your MySQL password
    "database": "sentiment_analysis"
}
```

**Note**: If you don't have MySQL, you can still test the app, but you'll need to train a model first (see Step 3).

### Step 3: Train Initial Model (REQUIRED)

The app needs a trained model to make predictions. You have two options:

#### Option A: Train with Database (Recommended)

1. **Set up MySQL database**:
```sql
CREATE DATABASE sentiment_analysis;
USE sentiment_analysis;

CREATE TABLE twitter_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    clean_text TEXT,
    category INT
);

-- Add some sample data
INSERT INTO twitter_data (clean_text, category) VALUES
('I love this product!', 1),
('This is amazing', 1),
('Terrible experience', 0),
('Not worth it', 0);
```

2. **Train the model**:
```bash
python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"
```

#### Option B: Quick Test Without Database

If you just want to test the app quickly without setting up a database:

1. **Create a simple test script** (`quick_train.py`):
```python
from ml.models import MODELS
from ml.preprocessing import create_vectorizer
from ml.evaluator import evaluate
from sklearn.model_selection import train_test_split
import pickle
from config import VECTORIZER_PATH, CURRENT_MODEL_PATH
import pandas as pd

# Create simple test data
data = {
    'clean_text': [
        'I love this', 'This is great', 'Amazing product',
        'Terrible', 'Bad experience', 'Not good',
        'Excellent', 'Wonderful', 'Fantastic',
        'Awful', 'Poor quality', 'Disappointing'
    ],
    'category': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['category'], test_size=0.2, random_state=42
)

# Create vectorizer
vectorizer, X_train_vec, X_test_vec = create_vectorizer(X_train, X_test)

# Train and evaluate
best_name, best_model, best_acc = evaluate(MODELS, X_train_vec, y_train, X_test_vec, y_test)

# Save vectorizer
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

# Save model
model_data = {
    "model": best_model,
    "model_name": best_name,
    "accuracy": best_acc
}
with open(CURRENT_MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"Model trained: {best_name} with accuracy: {best_acc:.4f}")
```

2. **Run the training script**:
```bash
python quick_train.py
```

### Step 4: Run the Application

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
```

### Step 5: Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

or

```
http://127.0.0.1:5000
```

##  Using the Application

1. **Enter text** in the textarea (e.g., "I love this product!")
2. **Click "Analyze Sentiment"**
3. **View results**:
   - Prediction (Positive/Negative)
   - Confidence score
   - Model information

### Optional: Enable Self-Learning

- Check the "Enable self-learning" checkbox
- High-confidence predictions will be added to training data
- Requires database connection

##  Troubleshooting

### Error: "Model file not found"

**Solution**: Train a model first (Step 3)

```bash
python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"
```

### Error: "Database connection failed"

**Solutions**:
1. Make sure MySQL is running
2. Check credentials in `config.py`
3. Or use Option B above to test without database

### Error: "Port 5000 already in use"

**Solution**: Change the port in `app.py`:

```python
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Use different port
```

Or find and kill the process using port 5000:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### Error: "Module not found"

**Solution**: Make sure you're in the virtual environment and dependencies are installed:

```bash
# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

##  Common Commands

```bash
# Run the app
python app.py

# Train a new model
python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"

# Check system info via API
curl http://localhost:5000/api/system_info

# View logs
tail -f logs/system.log  # Linux/Mac
type logs\system.log     # Windows
```

##  What You'll See

When you run the app, you'll see:

1. **System Information Panel**:
   - Model version
   - Algorithm name
   - Accuracy
   - Last drift score

2. **Prediction Form**:
   - Text input area
   - Self-learning toggle
   - Analyze button

3. **Results** (after prediction):
   - Sentiment (Positive/Negative)
   - Confidence bar
   - Model information

##  Next Steps

- **Enable scheduler** for automated retraining: `python scheduler.py`
- **Check logs**: `logs/system.log`
- **View model metadata**: `models_store/metadata.json`
- **Read full documentation**: See [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)

---

**Need more help?** Check the [full documentation](README.md) or [troubleshooting guide](DEPLOYMENT.md#troubleshooting).
