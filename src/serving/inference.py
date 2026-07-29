"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================
"""

import os
import sys
import pandas as pd
import mlflow

# === DYNAMIC PATH CONFIGURATION ===
# Automatically locates the project root relative to this file's location (src/serving/inference.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

# DB sits in the project root by default
DB_PATH = os.path.join(REPO_ROOT, "mlflow.db")

# Point MLflow to SQLite database
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

model = None
latest_run_id = None

# Load latest model from MLflow DB
try:
    experiment = mlflow.get_experiment_by_name("Telco Churn - XGBoost")
    
    if experiment is None:
        raise Exception("Experiment 'Telco Churn - XGBoost' not found in SQLite DB.")
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["attribute.start_time DESC"]
    )
    
    if runs.empty:
        raise Exception("No runs found inside the SQLite DB for this experiment.")
        
    latest_run_id = runs.iloc[0]["run_id"]
    MODEL_URI = f"runs:/{latest_run_id}/model"
    
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"✅ Successfully loaded model from SQLite (Run ID: {latest_run_id})")

except Exception as e:
    raise Exception(f"❌ Failed to load model from SQLite DB: {e}")

# === FEATURE SCHEMA LOADING ===
FEATURE_COLS = []
try:
    # 1. Try reading directly from local serving directory artifacts dynamically
    local_feat_path = os.path.join(
        REPO_ROOT, "src", "serving", "model", latest_run_id, "artifacts", "feature_columns.txt"
    )
    
    if os.path.exists(local_feat_path):
        with open(local_feat_path, "r") as f:
            FEATURE_COLS = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from local file: {local_feat_path}")
        
    # 2. Try MLflow Signature if local file missing
    elif hasattr(model, "metadata") and model.metadata.signature and model.metadata.signature.inputs:
        FEATURE_COLS = model.metadata.signature.inputs.input_names()
        print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from MLflow Signature")

    if not FEATURE_COLS:
        raise Exception("Could not find feature_columns.txt or MLflow signature.")

except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )
    
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    
    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"
