import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from .data_loader import load_data
from .preprocessing import clean_data, add_business_features, split_xy, build_preprocessor
from .config import MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, MODEL_DIR, REPORT_DIR, ID_COL


def train_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_data()
    df = add_business_features(clean_data(raw_df))
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.22, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced", max_depth=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    trained = {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1_score": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, prob), 4)
        })
        trained[name] = pipe

    best_row = sorted(results, key=lambda x: (x["roc_auc"], x["recall"], x["f1_score"]), reverse=True)[0]
    best_model = trained[best_row["model"]]
    joblib.dump(best_model, MODEL_PATH)

    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_pred = (best_prob >= 0.50).astype(int)
    pred_df = raw_df.loc[idx_test, [ID_COL]].copy()
    pred_df["actual_churn"] = y_test.values
    pred_df["churn_probability"] = best_prob
    pred_df["risk_level"] = pd.cut(
        pred_df["churn_probability"], bins=[0, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"], include_lowest=True
    )
    pred_df["recommended_action"] = pred_df["risk_level"].map({
        "Low": "Maintain engagement",
        "Medium": "Send personalized offer",
        "High": "Priority retention call + discount"
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "best_model": best_row["model"],
        "leaderboard": results,
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "classification_report": classification_report(y_test, best_pred, output_dict=True)
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=4))
    return metrics, pred_df


if __name__ == "__main__":
    metrics, _ = train_models()
    print(json.dumps(metrics, indent=2))
