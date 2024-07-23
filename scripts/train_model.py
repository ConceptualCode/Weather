import mlflow
import mlflow.sklearn
import pandas as pd
import json
import os
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


def train_model(data_path, model_type):
    # # Set the tracking URI
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # # Create a new experiment or get existing one
    # experiment_name = "Weather Data Training"
    # try:
    #     experiment_id = mlflow.create_experiment(experiment_name)
    #     print(f"Created new experiment with ID: {experiment_id}")
    # except:
    #     experiment = mlflow.get_experiment_by_name(experiment_name)
    #     experiment_id = experiment.experiment_id
    #     print(f"Using existing experiment with ID: {experiment_id}")

    # with mlflow.start_run(experiment_id=experiment_id, run_name="Model Training"):
    with mlflow.start_run():
        print("MLflow run started")

        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_type", model_type)

        # Log environment information
        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.set_tag("python_version", os.popen('python --version').read().strip())

        # Load processed data
        df = pd.read_csv(data_path)
        print("Data loaded")

        # Log the dataset as an artifact
        mlflow.log_artifact(data_path, "data")
        print("Data artifact logged")

        # # Define features and target
        # target = 'temp'
        # features = df.drop(columns=[target])
        # #target = df['temp']

        # Define features and target
        features = df.drop(columns=['temp', 'date'])
        target = df['temp']

        # Convert integer columns to floats
        int_columns = features.select_dtypes(include=['int']).columns
        features[int_columns] = features[int_columns].astype('float')
        print("Converted integer columns to floats")

        # Identify categorical columns (assuming they are of type object)
        categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

        # Create a preprocessor for the categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Keep remaining columns as they are
        )

        # Define the model based on model_type
        if model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'LinearRegression':
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Create a pipeline that first applies the preprocessor and then trains the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])


        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        print("Data split into training and testing sets")

        # Log the train and test sets
        X_train.to_csv("data/X_train.csv", index=False)
        X_test.to_csv("data/X_test.csv", index=False)
        y_train.to_csv("data/y_train.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)
        mlflow.log_artifact("data/X_train.csv", "data")
        mlflow.log_artifact("data/X_test.csv", "data")
        mlflow.log_artifact("data/y_train.csv", "data")
        mlflow.log_artifact("data/y_test.csv", "data")
        print("Train and test sets logged as artifacts")

        # Train the model
        pipeline.fit(X_train, y_train)
        print("Model trained")

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        print("Metrics logged")

        # Log the model signature
        input_example = X_test.iloc[:5]
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, "model", signature=signature, input_example=input_example)
        print("Model logged")

        print(f"Model trained and logged with MAE: {mae}, MSE: {mse}, R²: {r2}, RMSE: {rmse}")
        print("MLflow run completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and train a weather data model")
    parser.add_argument('--data_path', type=str, help="Path to the raw weather data CSV file")
    parser.add_argument('--model_type', type=str, default='RandomForest', choices=['RandomForest', 'LinearRegression'], help="Type of model to train")
    args = parser.parse_args()
    train_model(args.data_path, args.model_type)