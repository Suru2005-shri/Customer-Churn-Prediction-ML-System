from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.preprocessing import add_business_features
from src.config import MODEL_PATH

app = FastAPI(title="Customer Churn Prediction API", version="1.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class CustomerInput(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.0
    TotalCharges: float = 1020.0

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/score")
def score_customer(customer: CustomerInput):
    if model is None:
        return {"error": "Model not found. Run python main.py first."}
    df = pd.DataFrame([customer.model_dump()])
    df = add_business_features(df)
    prob = float(model.predict_proba(df)[0][1])
    risk = "High" if prob >= 0.60 else "Medium" if prob >= 0.30 else "Low"
    action = {
        "High": "Immediate retention call + personalized discount",
        "Medium": "Send targeted offer + usage education email",
        "Low": "Maintain engagement and loyalty rewards"
    }[risk]
    return {"churn_probability": round(prob, 4), "risk_level": risk, "recommended_action": action}
