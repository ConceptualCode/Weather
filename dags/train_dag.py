from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1
}

def run_training():
    subprocess.run(["python3", "scripts/train_model.py", "--data_path", "data/processed_weather_data.csv", "--model_type", "RandomForest"])

dag = DAG('train_model', default_args=default_args, schedule_interval='@weekly')

wait_for_preprocessing = ExternalTaskSensor(
    task_id='wait_for_preprocessing',
    external_dag_id='data_preprocessing',
    external_task_id='preprocess_data',
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=run_training,
    dag=dag
)

wait_for_preprocessing >> train_task