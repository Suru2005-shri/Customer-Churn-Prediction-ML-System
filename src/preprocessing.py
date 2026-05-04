import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from .config import TARGET, ID_COL


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_value"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"]
    )
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_online_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["OnlineBackup"] == "Yes") |
        (df["TechSupport"] == "Yes")
    ).astype(int)
    df["premium_service_count"] = (
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["monthly_charge_per_service"] = df["MonthlyCharges"] / (df["premium_service_count"] + 1)
    df["risk_payment_flag"] = df["PaymentMethod"].str.contains("Electronic check", case=False, na=False).astype(int)
    return df


def split_xy(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
