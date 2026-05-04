from src.model_training import train_models
from src.config import MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH


if __name__ == "__main__":
    print("🚀 Starting Customer Churn Model Training...\n")

    metrics, predictions = train_models()

    print("\n✅ Training Completed Successfully!")
    print("🏆 Best Model:", metrics["best_model"])

    print("\n📊 Model Performance Summary:")
    for model in metrics["leaderboard"]:
        print(
            f"{model['model']} | "
            f"Accuracy: {model['accuracy']} | "
            f"Recall: {model['recall']} | "
            f"F1: {model['f1_score']} | "
            f"ROC-AUC: {model['roc_auc']}"
        )

    print("\n📁 Files Generated:")
    print(f"✔ Model saved at → {MODEL_PATH}")
    print(f"✔ Metrics saved at → {METRICS_PATH}")
    print(f"✔ Predictions saved at → {PREDICTIONS_PATH}")

    print("\n🎯 You can now run the dashboard:")
    print("👉 streamlit run app.py")