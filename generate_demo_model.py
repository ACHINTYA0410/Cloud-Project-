"""
generate_demo_model.py
──────────────────────
Generates a realistic isolation_forest.pkl + scaler.pkl + model_metadata.json
using synthetic data so the Flask backend can start immediately without the
full UCI dataset.

Run once before starting app.py:
    python generate_demo_model.py
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

np.random.seed(42)
N = 50_000

# Generate realistic synthetic readings matching UCI household power ranges
data = np.column_stack([
    np.random.uniform(0.2, 6.0, N),        # Global_active_power
    np.random.uniform(0.0, 1.0, N),        # Global_reactive_power
    np.random.uniform(225.0, 243.0, N),    # Voltage
    np.random.uniform(0.2, 25.0, N),       # Global_intensity
    np.random.uniform(0.0, 20.0, N),       # Sub_metering_1
    np.random.uniform(0.0, 10.0, N),       # Sub_metering_2
    np.random.uniform(0.0, 18.0, N),       # Sub_metering_3
])

# Inject 1% anomaly spikes
anomaly_idx = np.random.choice(N, size=int(N * 0.01), replace=False)
data[anomaly_idx, 0] *= 4.0   # power spike
data[anomaly_idx, 3] *= 3.5   # intensity spike

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

model = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
model.fit(X_scaled)

joblib.dump(model, "isolation_forest.pkl")
joblib.dump(scaler, "scaler.pkl")

means = dict(zip(FEATURES, data.mean(axis=0).tolist()))
stds  = dict(zip(FEATURES, data.std(axis=0).tolist()))

metadata = {
    "model_type": "Isolation Forest",
    "contamination": 0.01,
    "n_estimators": 100,
    "features": FEATURES,
    "total_records_trained": N,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "feature_means": means,
    "feature_stds":  stds,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Demo model trained on {N:,} synthetic records.")
print("Files written: isolation_forest.pkl, scaler.pkl, model_metadata.json")
