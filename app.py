"""
app.py — Power Anomaly Detection Flask Backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import requests as http_requests
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "ngrok-skip-browser-warning"])

# ── Feature names (must match Supabase column names exactly) ───────────────────
FEATURES = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

# ── Model ─────────────────────────────────────────────────────────────────────
_model  = None
_scaler = None

def load_model():
    """Load model eagerly at startup. Raises if files are missing."""
    global _model, _scaler
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base_dir, "isolation_forest.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"isolation_forest.pkl not found at {model_path}.\n"
            "Run train_model.py or generate_demo_model.py first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"scaler.pkl not found at {scaler_path}.\n"
            "Run train_model.py or generate_demo_model.py first."
        )

    _model  = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    return _model, _scaler


# ── Supabase helpers ───────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()


def _supabase_ready() -> bool:
    return (
        bool(SUPABASE_URL)
        and bool(SUPABASE_KEY)
        and "your-project-id" not in SUPABASE_URL
        and "your-service-role-key" not in SUPABASE_KEY
    )


def _get_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def supabase_insert(rows: list[dict]) -> list:
    """Insert rows into the predictions table. Returns inserted rows or []."""
    if not _supabase_ready():
        print("[Supabase] ⚠️  Credentials not configured — skipping insert.")
        return []
    try:
        url  = f"{SUPABASE_URL}/rest/v1/predictions"
        resp = http_requests.post(url, headers=_get_headers(), json=rows, timeout=10)
        if resp.ok:
            return resp.json()
        else:
            print(f"[Supabase] ❌ Insert failed {resp.status_code}: {resp.text}")
            return []
    except Exception as e:
        print(f"[Supabase] ❌ Insert error: {e}")
        return []


def supabase_query(filters: str = "", limit: int = 100, offset: int = 0) -> list:
    if not _supabase_ready():
        return []
    try:
        url  = (
            f"{SUPABASE_URL}/rest/v1/predictions"
            f"?order=timestamp.desc&limit={limit}&offset={offset}{filters}"
        )
        resp = http_requests.get(url, headers=_get_headers(), timeout=10)
        return resp.json() if resp.ok else []
    except Exception as e:
        print(f"[Supabase] ❌ Query error: {e}")
        return []


def supabase_count(filter_anomaly=None) -> int:
    if not _supabase_ready():
        return 0
    try:
        headers    = {**_get_headers(), "Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"}
        filter_str = ""
        if filter_anomaly is not None:
            filter_str = f"&is_anomaly=eq.{str(filter_anomaly).lower()}"
        url  = f"{SUPABASE_URL}/rest/v1/predictions?select=id{filter_str}"
        resp = http_requests.get(url, headers=headers, timeout=10)
        total = resp.headers.get("content-range", "0/0").split("/")[-1]
        return int(total) if total.isdigit() else 0
    except Exception as e:
        print(f"[Supabase] ❌ Count error: {e}")
        return 0


def classify_deviation(is_anomaly: bool, score: float) -> dict:
    """
    Technical triage labels for model outputs.
    - inlier
    - borderline_outlier
    - moderate_outlier
    - critical_outlier
    """
    if not is_anomaly:
        return {
            "anomaly_class": "inlier",
            "anomaly_severity": "none",
            "risk_index": 0.0,
        }

    model_offset = float(getattr(_model, "offset_", -0.5))
    margin = max(0.0, model_offset - float(score))

    if margin >= 0.060:
        anomaly_class = "critical_outlier"
        anomaly_severity = "high"
    elif margin >= 0.025:
        anomaly_class = "moderate_outlier"
        anomaly_severity = "medium"
    else:
        anomaly_class = "borderline_outlier"
        anomaly_severity = "low"

    # 0..1 risk index from model-distance to decision boundary
    risk_index = min(1.0, round(margin / 0.1, 4))

    return {
        "anomaly_class": anomaly_class,
        "anomaly_severity": anomaly_severity,
        "risk_index": risk_index,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":               "ok",
        "model":                "Isolation Forest",
        "model_loaded":         _model is not None,
        "cloud":                "AWS Elastic Beanstalk",
        "database":             "Supabase",
        "supabase_configured":  _supabase_ready(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data     = request.json
    # Accepts: { "readings": [[gap, grp, voltage, gi, sm1, sm2, sm3], ...] }
    readings = np.array(data["readings"], dtype=float)

    scaled = _scaler.transform(readings)
    preds  = _model.predict(scaled)         # -1 = anomaly, 1 = normal
    scores = _model.score_samples(scaled)   # lower = more anomalous

    results        = []
    rows_to_insert = []

    for i, (pred, score, reading) in enumerate(zip(preds, scores, readings)):
        is_anomaly = bool(pred == -1)
        timestamp  = datetime.now(timezone.utc).isoformat()
        triage = classify_deviation(is_anomaly, float(score))

        # Column names must match Supabase table exactly (all lowercase_underscore)
        row = {
            "timestamp":              timestamp,
            "global_active_power":    float(reading[0]),
            "global_reactive_power":  float(reading[1]),
            "voltage":                float(reading[2]),
            "global_intensity":       float(reading[3]),
            "sub_metering_1":         float(reading[4]),
            "sub_metering_2":         float(reading[5]),
            "sub_metering_3":         float(reading[6]),
            "anomaly_score":          float(score),
            "is_anomaly":             is_anomaly,
        }
        rows_to_insert.append(row)

        results.append({
            "index":     i,
            "timestamp": timestamp,
            "anomaly":   is_anomaly,
            "score":     float(score),
            **triage,
            "readings":  {k: float(reading[j]) for j, k in enumerate(FEATURES)},
        })

    inserted = supabase_insert(rows_to_insert)
    if inserted:
        print(f"[Supabase] ✅ Inserted {len(inserted)} row(s)")

    return jsonify(results)


@app.route("/history", methods=["GET"])
def history():
    page_size      = int(request.args.get("page_size", request.args.get("limit", 100)))
    page           = int(request.args.get("page", 1))
    anomalies_only = request.args.get("anomalies_only", "false").lower() == "true"
    page_size      = max(1, min(page_size, 500))
    page           = max(1, page)
    offset         = (page - 1) * page_size
    filters        = "&is_anomaly=eq.true" if anomalies_only else ""
    rows           = supabase_query(filters=filters, limit=page_size, offset=offset)
    total          = supabase_count(filter_anomaly=True) if anomalies_only else supabase_count()

    enriched_rows = []
    for row in rows:
        triage = classify_deviation(bool(row.get("is_anomaly")), float(row.get("anomaly_score", 0.0)))
        enriched_rows.append({**row, **triage})

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return jsonify({
        "rows": enriched_rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    })


@app.route("/stats", methods=["GET"])
def stats():
    total     = supabase_count()
    anomalies = supabase_count(filter_anomaly=True)
    normal    = total - anomalies
    return jsonify({
        "total_predictions": total,
        "total_anomalies":   anomalies,
        "normal_readings":   normal,
        "anomaly_rate":      round((anomalies / total * 100), 2) if total > 0 else 0,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(base_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "model_metadata.json not found. Run train_model.py first."}), 404
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return jsonify(metadata)


# ── Startup ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Power Anomaly Detection — Flask Backend")
    print("=" * 60)

    # Eager model load — fail fast if PKL files are missing
    try:
        load_model()
        print("  ✅ Model loaded:     isolation_forest.pkl + scaler.pkl")
    except FileNotFoundError as e:
        print(f"  ❌ Model load FAILED:\n     {e}")
        import sys
        sys.exit(1)

    # Supabase status
    if _supabase_ready():
        print(f"  ✅ Supabase:         {SUPABASE_URL}")
    else:
        print("  ⚠️  Supabase:         NOT configured (check backend/.env)")

    print("  🚀 Starting on      http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
