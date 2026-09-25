"""
StreamPulse Music Analytics - Training Script with MLflow Tracking.
Trains and evaluates the music stream forecasting pipeline.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add parent directory to path to enable importing features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import SpotifyFeatureCreator


def load_and_clean_data(csv_path):
    """Load raw dataset and filter invalid/corrupted records."""
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Convert streams to numeric and drop invalid non-numeric rows
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams']).reset_index(drop=True)

    # Clean comma-formatted numerical columns
    for col in ['in_deezer_playlists', 'in_shazam_charts']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def build_pipeline(alpha=10.0):
    """Build scikit-learn pipeline with custom feature engineering and model."""
    numeric_features = [
        'bpm', 'mode_binary', 'danceability_%', 'valence_%', 'energy_%',
        'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%',
        'track_age_years', 'release_month_sin', 'release_month_cos',
        'log_total_playlists', 'total_charts', 'dance_energy_ratio'
    ]
    categorical_features = ['key']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('creator', SpotifyFeatureCreator(reference_year=2023)),
            ('prep', preprocessor),
            ('model', Ridge(alpha=alpha, random_state=42))
        ]
    )
    return pipeline


def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'raw', 'spotify_2023.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading raw Spotify dataset from: {data_path}")
    df = load_and_clean_data(data_path)
    print(f"Dataset successfully loaded. Valid tracks: {len(df)}")

    # Target variable: log1p(streams) to handle wide orders of magnitude
    y = np.log1p(df['streams'])
    X = df.drop(columns=['streams'])

    # Strict train-only fitting discipline (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train split: {X_train.shape[0]} tracks | Test split: {X_test.shape[0]} tracks")

    # MLflow Tracking Setup
    mlflow.set_experiment("StreamPulse_Music_Forecasting")

    # Run 1: Baseline Linear Regression
    print("\n--- Training Experiment 1: Baseline Linear Regression ---")
    with mlflow.start_run(run_name="Baseline_Linear_Regression"):
        base_pipe = Pipeline([
            ('creator', SpotifyFeatureCreator()),
            ('prep', ColumnTransformer([
                ('num', StandardScaler(), [
                    'bpm', 'mode_binary', 'danceability_%', 'valence_%', 'energy_%',
                    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%',
                    'track_age_years', 'release_month_sin', 'release_month_cos',
                    'log_total_playlists', 'total_charts', 'dance_energy_ratio'
                ]),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['key'])
            ])),
            ('model', LinearRegression())
        ])
        base_pipe.fit(X_train, y_train)
        base_preds = base_pipe.predict(X_test)
        base_r2 = r2_score(y_test, base_preds)
        base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))
        
        mlflow.log_param("model_family", "LinearRegression")
        mlflow.log_metric("test_r2", base_r2)
        mlflow.log_metric("test_rmse_log", base_rmse)
        print(f"Baseline Linear Regression Test R2: {base_r2:.4f}, RMSE: {base_rmse:.4f}")

    # Run 2: Ridge Regressor with Regularization
    print("\n--- Training Experiment 2: Regularized Ridge Pipeline (Production Model) ---")
    with mlflow.start_run(run_name="Production_Ridge_Pipeline"):
        prod_pipe = build_pipeline(alpha=10.0)
        prod_pipe.fit(X_train, y_train)
        prod_preds = prod_pipe.predict(X_test)

        prod_r2 = r2_score(y_test, prod_preds)
        prod_rmse = np.sqrt(mean_squared_error(y_test, prod_preds))
        prod_mae = mean_absolute_error(y_test, prod_preds)

        # Log parameters & metrics to MLflow
        mlflow.log_param("model_family", "Ridge")
        mlflow.log_param("alpha", 10.0)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_metric("test_r2", prod_r2)
        mlflow.log_metric("test_rmse_log", prod_rmse)
        mlflow.log_metric("test_mae_log", prod_mae)

        # Save production pipeline with joblib
        output_model_path = os.path.join(models_dir, 'spotify_pipeline.joblib')
        joblib.dump(prod_pipe, output_model_path)
        print(f"Production pipeline saved to: {output_model_path}")
        print(f"Ridge Model Test R2: {prod_r2:.4f}, RMSE: {prod_rmse:.4f}, MAE: {prod_mae:.4f}")

    print("\nTraining and MLflow logging completed successfully!")
    return prod_pipe


if __name__ == '__main__':
    train()
