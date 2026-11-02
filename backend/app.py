from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "backend" / "models"
DATA = ROOT / "dataset" / "Crop_recommendation.csv"
with (MODELS / "model_metadata.json").open(encoding="utf-8") as f:
    META = json.load(f)
MODEL_NAME = META["best_model"]
MODEL = joblib.load(MODELS / "best_model.pkl")
FEATURES = META["feature_names"]

app = Flask(__name__, template_folder=str(ROOT / "frontend" / "templates"),
            static_folder=str(ROOT / "frontend" / "static"))

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/stats")
def stats():
    counts = {}
    if DATA.exists():
        df = pd.read_csv(DATA)
        counts = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
    rows = []
    for name, m in META["models"].items():
        rows.append({"name": name, "accuracy": m["accuracy"],
                     "precision_macro": m["precision_macro"],
                     "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                     "cv_f1_macro_mean": m["cv_f1_macro_mean"]})
    best = META["models"][MODEL_NAME]
    return jsonify({
        "dataset_rows": META["dataset_rows"], "class_count": len(META["class_names"]),
        "class_names": META["class_names"], "class_counts": counts,
        "best_model": MODEL_NAME, "models": rows,
        "best_metrics": {k: best[k] for k in
                         ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
        "confusion_matrix": best["confusion_matrix"],
        "selection_metric": META["selection_metric"]
    })

@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send the input values as JSON."), 400
    values = []
    for feature in FEATURES:
        try:
            value = float(data.get(feature))
        except (TypeError, ValueError):
            return jsonify(error=f"{feature} must be a valid number."), 400
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            return jsonify(error=f"{feature} must be a finite number."), 400
        values.append(value)
    frame = pd.DataFrame([values], columns=FEATURES)
    crop = str(MODEL.predict(frame)[0])
    top = []
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(frame)[0]
        ranked = sorted(zip(MODEL.classes_, probs), key=lambda x: float(x[1]), reverse=True)[:3]
        top = [{"crop": str(label), "probability": float(prob)} for label, prob in ranked]
    return jsonify(crop=crop, model=MODEL_NAME, top_predictions=top)

if __name__ == "__main__":
    print(f"CropSense running at http://127.0.0.1:5000 (model: {MODEL_NAME})")
    app.run(host="127.0.0.1", port=5000, debug=False)
