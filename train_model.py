"""
train_model.py
──────────────
Trains an Isolation Forest on the UCI Individual Household Electric
Power Consumption dataset and saves the model artifacts.

Run from the backend/ directory:
    python train_model.py

Output files (written to backend/):
    isolation_forest.pkl
    scaler.pkl
    model_metadata.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Dataset path ──────────────────────────────────────────────────────────────
# The CSV sits one level up from backend/ inside the dataset folder.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(
    BASE_DIR, "..", "individual+household+electric+power+consumption"
)
DATASET_PATH = os.path.join(DATASET_DIR, "household_power_consumption.txt")

# ── Config ────────────────────────────────────────────────────────────────────
NROWS = None             # Set to None for all ~2M rows.
CONTAMINATION = 0.01     # Expected fraction of anomalies
N_ESTIMATORS  = 100
RANDOM_STATE  = 42

# ── Exact feature columns from the UCI dataset ────────────────────────────────
FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"[1/5] Loading dataset from:\n      {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at:\n  {DATASET_PATH}\n"
        "Make sure the folder 'individual+household+electric+power+consumption' "
        "exists in the project root."
    )

df = pd.read_csv(
    DATASET_PATH,
    sep=";",                  # Dataset uses semicolons as delimiter
    na_values=["?"],          # Missing values are encoded as '?'
    low_memory=False,
    nrows=NROWS,
)

print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── Parse datetime (not used in model but good for metadata) ──────────────────
print("[2/5] Cleaning data…")
if "Date" in df.columns and "Time" in df.columns:
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df.drop(["Date", "Time"], axis=1, inplace=True)

# Drop rows with any NaN in the feature columns
before = len(df)
df.dropna(subset=FEATURES, inplace=True)
after  = len(df)
print(f"      Dropped {before - after:,} rows with missing values. "
      f"{after:,} clean rows remain.")

# ── Feature matrix ────────────────────────────────────────────────────────────
X = df[FEATURES].astype(float).values
print(f"[3/5] Feature matrix shape: {X.shape}")

# ── Scale + Train ─────────────────────────────────────────────────────────────
print("[4/5] Training Isolation Forest…")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,             # Use all CPU cores
)
model.fit(X_scaled)
print("      Training complete.")

# ── Save artifacts ────────────────────────────────────────────────────────────
print("[5/5] Saving model artifacts…")
joblib.dump(model,  os.path.join(BASE_DIR, "isolation_forest.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

X_df = pd.DataFrame(X, columns=FEATURES)
metadata = {
    "model_type":             "Isolation Forest",
    "contamination":          CONTAMINATION,
    "n_estimators":           N_ESTIMATORS,
    "random_state":           RANDOM_STATE,
    "features":               FEATURES,
    "total_records_trained":  len(X),
    "training_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "feature_means":          X_df.mean().round(6).to_dict(),
    "feature_stds":           X_df.std().round(6).to_dict(),
}
with open(os.path.join(BASE_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("✅ Done!")
print(f"   isolation_forest.pkl  →  {os.path.join(BASE_DIR, 'isolation_forest.pkl')}")
print(f"   scaler.pkl            →  {os.path.join(BASE_DIR, 'scaler.pkl')}")
print(f"   model_metadata.json   →  {os.path.join(BASE_DIR, 'model_metadata.json')}")
print(f"   Trained on {len(X):,} clean records.")
