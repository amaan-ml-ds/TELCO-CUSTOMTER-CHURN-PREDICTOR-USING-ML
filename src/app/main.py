import sys
import os

# Ensure repo root is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.serving.inference import predict

def main():
    print("🚀 Initializing Inference Pipeline Test...")

    # Sample payload for testing prediction
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    try:
        prediction_result = predict(sample_customer)
        print("\n========================================")
        print(f"🔮 Prediction Result: {prediction_result}")
        print("========================================\n")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()
