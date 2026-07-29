import os
import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from mlflow.models import infer_signature

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model, logs metrics/params to MLflow under 'Telco Churn - XGBoost',
    and saves feature artifacts.
    """
    # 🔑 Ensure experiment name is explicitly set
    mlflow.set_experiment("Telco Churn - XGBoost")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        signature = infer_signature(X_train, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model", 
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        # Save local feature_columns.txt artifact inside src/serving/model/<run_id>/artifacts/
        artifacts_dir = os.path.join("src", "serving", "model", run_id, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        feat_file = os.path.join(artifacts_dir, "feature_columns.txt")
        
        with open(feat_file, "w") as f:
            for col in X_train.columns:
                f.write(f"{col}\n")

        print(f"✅ Model trained successfully (Run ID: {run_id}). Accuracy: {acc:.4f}, Recall: {rec:.4f}")
