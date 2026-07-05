from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import get_current_context
from airflow.providers.standard.operators.python import PythonOperator
import mlflow
import pandas as pd

def run():
    context = get_current_context()
    var = context["var"]
    mlflow1 = var["json"].get("mlflow")
    tracking_uri=mlflow1["tracking_uri"]
    model_run_id = mlflow1["model_run_id"]
    
    mlflow.set_tracking_uri(tracking_uri)
    
    model_uri = f"runs:/{model_run_id}/fraud"
    local_model_path = mlflow.artifacts.download_artifacts(model_uri)
    model = mlflow.sklearn.load_model(local_model_path)

    hook = PostgresHook(postgres_conn_id="postgres_default")
    conn = hook.get_conn()

    df = pd.read_sql("""
        SELECT id, caller_age_days, call_duration_sec, avg_call_duration_sec, fraud_label
        FROM fraud_data
    """, conn)

    X = df[
        ["caller_age_days", "call_duration_sec", "avg_call_duration_sec"]
    ]

    y_pred = model.predict(X)

    df["predicted"] = y_pred

    cur = conn.cursor()

    for row in df.itertuples():
        cur.execute("""
            UPDATE fraud_data
            SET predicted_fraud = %s
            WHERE id = %s
        """, (int(row.predicted), int(row.id)))

    conn.commit()
    cur.close()
    conn.close()


default_args = {
    "depends_on_past": False,
}

with DAG(
    dag_id="fraud_classifier_pg",
    default_args=default_args,
    description="Fraud classification using MLflow model",
    schedule=timedelta(days=1),
    start_date=datetime(2026, 6, 25),
    catchup=False,
    tags=["airflow"],
) as dag:

    predict_task = PythonOperator(
        task_id="predict2",
        python_callable=run,
    )
    