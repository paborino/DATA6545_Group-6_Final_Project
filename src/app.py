
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "champion_lightgbm_pipeline.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "champion_threshold.txt"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"error": "No JSON payload provided"}), 400

        if isinstance(payload, dict):
            input_df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            input_df = pd.DataFrame(payload)
        else:
            return jsonify({"error": "Payload must be a JSON object or list of objects"}), 400

        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        results = []
        for prob, pred in zip(probabilities, predictions):
            results.append({
                "high_severity_probability": float(prob),
                "predicted_high_severity": int(pred),
                "threshold": threshold
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
