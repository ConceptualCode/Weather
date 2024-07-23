import pandas as pd
import mlflow
import mlflow.pyfunc

def preprocess_data(input_path, output_path):
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # # Create or set the experiment
    # experiment_name = "Weather Data Preprocessing"
    # experiment_id = mlflow.set_experiment(experiment_name)
    # print(f"Experiment ID: {experiment_id}")

    with mlflow.start_run(run_name="Data Preprocessing"):
        #Log parameters
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_path", output_path)
        # Load the data
        data = pd.read_csv(input_path)

        # Convert temperature from Kelvin to Celsius
        data['temp'] = data['temp'] - 273.15
        data['temp_min'] = data['temp_min'] - 273.15
        data['temp_max'] = data['temp_max'] - 273.15

        # Convert date columns to datetime
        data['date'] = pd.to_datetime(data['date'])
        data['sunrise'] = pd.to_datetime(data['sunrise'])
        data['sunset'] = pd.to_datetime(data['sunset'])

        # Feature engineering: extract additional features from date
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['dayofweek'] = data['date'].dt.dayofweek
        #data['hour'] = data['date'].dt.hour

        # Drop columns that are not needed for modeling
        data = data.drop(columns=['country', 'city','latitude', 'longitude', 'sunrise', 'sunset', 'timezone', 'description', 'region'])

        data = data.dropna()
        # Save the processed data
        data.to_csv(output_path, index=False)

        # Log total lenght of the data
        total_lenght = len(data)
        mlflow.log_metric("total_length", total_lenght)

        # Log the preprocessed data artifact
        mlflow.log_artifact(output_path)

        print(f'Total length of the dataset: {total_lenght}')

if __name__ == "__main__":
    preprocess_data('data/nigeria_cities_weather_data.csv', 'data/processed_weather_data.csv')