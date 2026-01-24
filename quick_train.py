"""
Quick Training Script - Creates a simple model for testing without database
Run this before starting the app if you don't have a database set up.
"""

from ml.models import MODELS
from ml.preprocessing import create_vectorizer
from ml.evaluator import evaluate
from sklearn.model_selection import train_test_split
import pickle
from config import VECTORIZER_PATH, CURRENT_MODEL_PATH
import pandas as pd
from utils.model_lifecycle import ModelLifecycleManager

print("Creating test dataset...")

# Create simple test data
data = {
    'clean_text': [
        'I love this product', 'This is great', 'Amazing experience',
        'Wonderful service', 'Excellent quality', 'Fantastic',
        'Terrible product', 'Bad experience', 'Not good',
        'Awful service', 'Poor quality', 'Disappointing',
        'Very happy', 'Satisfied customer', 'Highly recommend',
        'Waste of money', 'Regret buying', 'Not recommended'
    ],
    'category': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)

print(f"Dataset created: {len(df)} samples")
print("Splitting data...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
)

print("Creating vectorizer...")
# Create vectorizer
vectorizer, X_train_vec, X_test_vec = create_vectorizer(X_train, X_test)

print("Training models...")
# Train and evaluate
best_name, best_model, best_acc = evaluate(MODELS, X_train_vec, y_train, X_test_vec, y_test)

print(f"Best model: {best_name} with accuracy: {best_acc:.4f}")

# Save vectorizer
print("Saving vectorizer...")
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

# Save model using lifecycle manager
print("Saving model...")
lifecycle_manager = ModelLifecycleManager()
version = lifecycle_manager.save_new_model(
    model=best_model,
    model_name=best_name,
    accuracy=best_acc,
    dataset_size=len(df),
    drift_score=None
)

# Also save as current_model.pkl for compatibility
model_data = {
    "model": best_model,
    "model_name": best_name,
    "accuracy": best_acc
}
with open(CURRENT_MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model trained and saved successfully!")
print(f"   Model: {best_name}")
print(f"   Version: v{version}")
print(f"   Accuracy: {best_acc:.4f}")
print(f"\nYou can now run: python app.py")
