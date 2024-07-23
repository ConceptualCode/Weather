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

def run_retraining():
    subprocess.run(["python3", "scripts/train_model.py", "--data_path", "data/processed_weather_data.csv", "--model_type", "RandomForest"])

dag = DAG('retrain_model', default_args=default_args, schedule_interval='@monthly')

wait_for_initial_training = ExternalTaskSensor(
    task_id='wait_for_initial_training',
    external_dag_id='model_training',
    external_task_id='train_model',
    mode='poke',  # can also use 'reschedule' if you want to use fewer resources
    timeout=600,  # timeout in seconds
    poke_interval=60,  # check every 60 seconds
    dag=dag
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=run_retraining,
    dag=dag
)

wait_for_initial_training >> retrain_task
