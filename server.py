from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("eth_fee_model_v3.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Body JSON:
    {
      "features": [f0, f1, ..., f7]
    }
    zwraca:
    [ fee+10, fee+20, ..., fee+60 ]  # wszystkie w Gwei
    """
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "JSON must contain key 'features'"}), 400

    feats = data["features"]
    if not isinstance(feats, list) or len(feats) != 8:
        return jsonify({"error": "features must be list[8]"}), 400

    try:
        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        preds = model.predict(x)[0].tolist() # 6 predykcji
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
