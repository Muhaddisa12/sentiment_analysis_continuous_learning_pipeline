import pickle, os
from datetime import date
from config import MODEL_DIR, CURRENT_MODEL_PATH

os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, name, acc):
    today = date.today().isoformat()
    daily_path = f"{MODEL_DIR}/{today}.pkl"

    payload = {
        "model": model,
        "model_name": name,
        "accuracy": acc
    }

    # Save daily model
    with open(daily_path, "wb") as f:
        pickle.dump(payload, f)

    # Save deployed model
    with open(CURRENT_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
