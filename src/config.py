from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "customer_churn_predictions.csv"

TARGET = "Churn"
ID_COL = "customerID"
