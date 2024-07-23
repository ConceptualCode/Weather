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

def run_preprocessing():
    subprocess.run(["python3", "scripts/data_preprocessing.py", "--input_path", "data/nigeria_cities_weather_data.csv", "--output_path", "data/processed_weather_data.csv"])

dag = DAG('data_preprocessing', default_args=default_args, schedule_interval='@daily')

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing,
    dag=dag
)