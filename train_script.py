import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:8070")

df = pd.read_csv("data_to_train.csv")

X = df[["caller_age_days", "call_duration_sec", "avg_call_duration_sec"]]
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

params = {
    "max_depth": 2,
    "min_samples_split": 1000,
    "random_state": 42,
}

with mlflow.start_run() as run:

    mlflow.log_params(params)

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="fraud",
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.set_tag("Training Info", "Basic DT model for fraud classifier")

    print("RUN ID =", run.info.run_id)
