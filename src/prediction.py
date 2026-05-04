import joblib
import pandas as pd
from .preprocessing import clean_data, add_business_features
from .config import MODEL_PATH, ID_COL


def load_model():
    return joblib.load(MODEL_PATH)


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    original = df.copy()
    working = add_business_features(clean_data(df)) if "Churn" in df.columns else add_business_features(df)
    X = working.drop(columns=["Churn", ID_COL], errors="ignore")
    proba = model.predict_proba(X)[:, 1]
    result = original[[ID_COL]].copy() if ID_COL in original.columns else pd.DataFrame({ID_COL: range(len(original))})
    result["churn_probability"] = proba
    result["risk_level"] = pd.cut(proba, bins=[0, 0.30, 0.60, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
    result["recommended_action"] = result["risk_level"].map({
        "Low": "Maintain engagement",
        "Medium": "Personalized retention offer",
        "High": "Immediate success manager call"
    })
    return result
