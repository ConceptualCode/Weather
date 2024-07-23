import mlflow.pyfunc
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_predict(data_path, run_id, model_name):
    # Construct the model URI
    model_uri = f"runs:/{run_id}/{model_name}"
    print(f"Model URI: {model_uri}")

    # Load the model from MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the data
    df = pd.read_csv(data_path)

    # Define features (assuming 'temp' is the target and 'date' is not needed)
    # features = df.drop()

    # Convert integer columns to floats if necessary
    int_columns = df.select_dtypes(include=['int']).columns
    df[int_columns] = df[int_columns].astype('float')

    # Make predictions
    predictions = model.predict(df)
    
    return predictions

if __name__ == "__main__":
    import argparse
import mlflow.pyfunc
import pandas as pd

def load_and_predict(data_path, run_id, model_name):
    # Construct the model URI
    model_uri = f"runs:/{run_id}/{model_name}"
    print(f"Model URI: {model_uri}")

    # Load the model from MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the data
    df = pd.read_csv(data_path)

    # Convert integer columns to floats if necessary
    int_columns = df.select_dtypes(include=['int']).columns
    df[int_columns] = df[int_columns].astype('float')

    # Make predictions
    predictions = model.predict(df)
    
    return predictions

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def plot_results(true_values, predictions, output_path):
    # Plotting the predictions vs true values (Subset)
    plt.figure(figsize=(10, 6))
    subset = 500  # Plot a subset to reduce clutter
    plt.scatter(range(subset), true_values[:subset], label='True Values', alpha=0.7, s=10)
    plt.scatter(range(subset), predictions[:subset], label='Predictions', alpha=0.7, s=10)
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature')
    plt.title('Predictions vs True Values (Subset)')
    plt.savefig(output_path.replace(".png", "_subset.png"))
    print(f"Subset plot saved to {output_path.replace('.png', '_subset.png')}")

    # Plotting the error distribution
    errors = true_values - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(output_path.replace(".png", "_errors.png"))
    print(f"Error distribution plot saved to {output_path.replace('.png', '_errors.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a trained model and make predictions on a dataset.")
    parser.add_argument('--data_path', type=str, help="Path to the dataset CSV file.")
    parser.add_argument('--run_id', type=str, help="MLflow run ID of the trained model.")
    parser.add_argument('--model_name', type=str, help="Name of the model in MLflow.")
    parser.add_argument('--true_values_path', type=str, help="Path to the true values CSV file.")
    parser.add_argument('--output_plot_path', type=str, default='predictions_vs_true.png', help="Path to save the output plot.")

    args = parser.parse_args()

    predictions = load_and_predict(args.data_path, args.run_id, args.model_name)
    true_values = pd.read_csv(args.true_values_path)['temp'].values
    mae, mse, r2 = evaluate_predictions(true_values, predictions)
    print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
    print(predictions)

    plot_results(true_values, predictions, args.output_plot_path)