# -*- coding: utf-8 -*-
"""
FC SCAN — Flask API Backend
منطق الحساب محفوظ بدون أي تعديل من fc_scan_fixed-3.py
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Fix NumPy >= 1.20 ────────────────────────────────────────────────────────
for _a, _t in [('float', float), ('int', int), ('complex', complex),
               ('bool', bool), ('object', object), ('str', str)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

import pyrenn as pr

app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── تحميل النموذج ─────────────────────────────────────────────────────────────
net = None
scaler_X = None
scaler_y = None
LM_OK = False

try:
    csv_path = os.path.join(SCRIPT_DIR, "ann_regression_model_pyrenn.csv")
    net = pr.loadNN(csv_path)
    scaler_X = joblib.load(os.path.join(SCRIPT_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(SCRIPT_DIR, "scaler_y.pkl"))
    LM_OK = True
    print(f"[FC SCAN] Modèle ANN-LM chargé depuis: {SCRIPT_DIR}")
except Exception as ex:
    print(f"[FC SCAN] ERREUR chargement modèle: {ex}")

# ── Classes béton ─────────────────────────────────────────────────────────────
CONCRETE_CLASSES = [
    (0,   16,  "#E74C3C", "Très faible"),
    (16,  20,  "#E67E22", "Faible"),
    (20,  25,  "#F39C12", "Ordinaire"),
    (25,  30,  "#2ECC71", "Résistant"),
    (30, 999,  "#2E86AB", "Haute perf."),
]
EC_CLASSES = ["C16/20", "C20/25", "C25/30", "C30/37", "C35/45"]

def get_concrete_class(fc):
    for i, (lo, hi, color, label) in enumerate(CONCRETE_CLASSES):
        if lo <= fc < hi:
            ec = EC_CLASSES[i] if i < len(EC_CLASSES) else "C35/45+"
            return {"label": label, "color": color, "ec": ec, "index": i}
    last = CONCRETE_CLASSES[-1]
    return {"label": last[3], "color": last[2], "ec": "C35/45+", "index": len(CONCRETE_CLASSES)-1}

# ── predict_model() — نفس المنطق بالضبط من § 1 في fc_scan_fixed-3.py ─────────
def predict_model(R: float, V: float) -> dict:
    if not LM_OK:
        return {"error": "Aucun fichier modèle trouvé dans: " + SCRIPT_DIR}
    x = np.array([R, V]).reshape(1, -1)
    try:
        xs  = scaler_X.transform(x)
        P   = xs.T
        ys  = pr.NNOut(P, net)
        fc  = float(scaler_y.inverse_transform(
                    np.array(ys).reshape(-1, 1))[0, 0])
        det = f"Modèle ANN actif — fc={fc:.4f} MPa"
        return {"fc": fc, "detail": det, "model": "ANN — LM (Modèle actif)"}
    except Exception as ex:
        return {"error": str(ex)}


# ═══════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════
@app.route("/")
def home():
    return jsonify({
        "message": "FC SCAN API is running",
        "health": "/health",
        "predict": "/predict"
    })
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": LM_OK,
        "model": "ANN — LM (pyrenn)" if LM_OK else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: {"R": float, "V": float}
    Returns: {"fc": float, "detail": str, "model": str, "class": {...}}
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Corps JSON manquant"}), 400

    try:
        R = float(data["R"])
        V = float(data["V"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Paramètres invalides: {e}"}), 400

    result = predict_model(R, V)

    if "error" in result:
        return jsonify(result), 500

    result["class"] = get_concrete_class(result["fc"])
    return jsonify(result)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    POST /predict_batch
    Body: {"items": [{"R": float, "V": float, "element": str, "fcd": float}, ...]}
    Returns: {"results": [...]}
    """
    data = request.get_json(force=True)
    if not data or "items" not in data:
        return jsonify({"error": "items manquants"}), 400

    results = []
    for item in data["items"]:
        try:
            R = float(item["R"])
            V = float(item["V"])
        except (KeyError, TypeError, ValueError) as e:
            results.append({"error": str(e), "element": item.get("element", "")})
            continue

        pred = predict_model(R, V)
        if "error" in pred:
            results.append({"error": pred["error"], "element": item.get("element", "")})
            continue

        pred["class"] = get_concrete_class(pred["fc"])
        pred["element"] = item.get("element", "")
        fcd = item.get("fcd", 0)
        if fcd and fcd > 0:
            pred["conforme"] = pred["fc"] >= fcd
            pred["fcd"] = fcd
        results.append(pred)

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
