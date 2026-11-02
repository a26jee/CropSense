
"""Command-line crop prediction using the saved best model."""

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "backend" / "models" / "best_model.pkl"

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity", "ph", "rainfall"
]


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run: python train.py"
        )

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Prepare input data
    sample = pd.DataFrame([{
        "N": n,
        "P": p,
        "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }], columns=FEATURES)

    # Predict crop
    crop = model.predict(sample)[0]

    # Calculate model-estimated probability
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        confidence = float(max(probabilities))

    return crop, confidence


if __name__ == "__main__":

    # Example measurements
    result, confidence = predict_crop(
        90, 42, 43,
        20.88, 82.00, 6.50, 202.94
    )

    print("\n--- CropSense Prediction ---")
    print("Recommended crop:", result)

    if confidence is not None:
        print(f"Model-estimated probability: {confidence:.2%}")
    else:
        print("Probability estimates are unavailable.")
