import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

run_id = "90e7837a993b4e279adbaf1bc865b62a"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/fraud")

test_df = pd.read_csv("data_to_inference.csv")

X_test = test_df[["caller_age_days", "call_duration_sec", "avg_call_duration_sec"]]

y_pred = model.predict(X_test)

y_true = test_df["fraud_label"]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
