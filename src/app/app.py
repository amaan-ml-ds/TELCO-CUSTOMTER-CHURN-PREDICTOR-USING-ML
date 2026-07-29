from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Ensure repo root is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.serving.inference import predict

app = FastAPI(title="Telco Churn Inference API", version="1.0.0")

class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "Yes"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "Yes"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.get("/")
def root():
    return {"status": "ok", "message": "Telco Churn API is active"}

@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        result = predict(payload)
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
