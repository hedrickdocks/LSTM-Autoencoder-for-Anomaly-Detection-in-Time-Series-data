import os
import sys
import sqlite3
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.losses import MeanSquaredError

from Preprocessor import MODEL_FEATURE_COLUMNS, Preprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'lstm_autoencoder.h5'))
SCALER_PATH = os.getenv('SCALER_PATH', os.path.join(BASE_DIR, 'scaler.pkl'))
THRESHOLDS_PATH = os.getenv('THRESHOLDS_PATH', os.path.join(BASE_DIR, 'anomaly_thresholds.pkl'))
TIME_STEPS = 24


def find_default_test_data():
    """Auto-detect available test datasets."""
    env_path = os.getenv('TEST_DATA_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Autoencoder Test Files', 'Saturday_dataset.csv')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'test_data.db')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Autoencoder Test Files', 'Friday_dataset.csv')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated_energy_data.csv')),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def get_labeled_test_data(data_path=None, inject_anomalies=True):
    """Loads test data and optionally injects known anomalies for evaluation."""
    resolved_path = data_path or find_default_test_data()
    if resolved_path and not os.path.exists(resolved_path):
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', resolved_path))
        if os.path.exists(alt_path):
            resolved_path = alt_path

    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Test data file not found at '{resolved_path}'. "
            "Please provide a valid file via --data or set TEST_DATA_PATH."
        )

    print(f"Loading evaluation dataset from: {resolved_path}")
    if resolved_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
        conn = sqlite3.connect(resolved_path)
        df = pd.read_sql_query(
            "SELECT datetime, voltage, current, power, energy, power_factor FROM energy_data ORDER BY datetime ASC LIMIT 1000",
            conn
        )
        conn.close()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    else:
        df = pd.read_csv(resolved_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

    df = df[MODEL_FEATURE_COLUMNS[:5]]
    if len(df) > 1000:
        df = df.iloc[:1000].copy()

    n_samples = len(df)
    if n_samples < 150:
        raise ValueError(f"Test data is too short ({n_samples} rows). Need at least 150 rows.")

    labels = np.zeros(n_samples, dtype=int)
    if inject_anomalies:
        print("Note: Injecting in-memory synthetic power spikes at indices [30, 80, 140] to compute classification metrics.")
        for idx in [30, 80, 140]:
            if idx < n_samples:
                df.iloc[idx, df.columns.get_loc('power')] = 15000
                labels[idx] = 1
    else:
        print("Evaluating raw baseline data (0 injected anomalies).")

    return df, labels

def evaluate_model(data_path=None, inject_anomalies=True):
    print("--- Starting Evaluation ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return

    df, raw_labels = get_labeled_test_data(data_path, inject_anomalies=inject_anomalies)


    prep = Preprocessor()
    prep.scaler = scaler
    prep.is_fitted = True

    thresholds = None
    if os.path.exists(THRESHOLDS_PATH):

        try:
            thresholds = joblib.load(THRESHOLDS_PATH)
            print(f"Loaded calibrated thresholds: Point={thresholds.get('point_threshold'):.5f}, Collective={thresholds.get('collective_threshold'):.5f}")
        except Exception as e:
            print(f"Could not load thresholds: {e}")

    df = prep.add_cyclical_time_features(df)
    df = df[MODEL_FEATURE_COLUMNS]
    scaled_data = prep.transform(df)
    X_test, target_timestamps = prep.create_sequences_gap_aware(
        scaled_data,
        timestamps=df.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15
    )

    df_timestamps_list = list(df.index)
    timestamp_to_label = {df_timestamps_list[idx]: raw_labels[idx] for idx in range(len(df_timestamps_list))}
    y_true_point = np.array([timestamp_to_label.get(ts, 0) for ts in target_timestamps])

    # Sequence is considered anomalous if any timestep within its 24-step window was injected with an anomaly
    y_true_seq = np.array([
        1 if np.any(raw_labels[max(0, i):min(len(raw_labels), i + TIME_STEPS)] == 1) else 0
        for i in range(len(X_test))
    ])

    print(f"Test Sequences Shape: {X_test.shape}")
    print(f"Point Anomaly Targets: {np.sum(y_true_point)} / {len(y_true_point)}")
    print(f"Anomalous Sequences: {np.sum(y_true_seq)} / {len(y_true_seq)}")

    print("Running inference...")
    X_pred = model.predict(X_test, verbose=0)

    orig_rescaled = prep.inverse_transform(X_test[:, -1, :])
    recon_rescaled = prep.inverse_transform(X_pred[:, -1, :])
    baseline = np.maximum(recon_rescaled, 1.0)
    anomaly_scores = np.max(np.abs(orig_rescaled - recon_rescaled) / baseline, axis=1)

    fixed_threshold = 0.50
    y_pred_fixed = (anomaly_scores > fixed_threshold).astype(int)

    print(f"\n--- Point Anomaly Metrics (Final Step Deviation > {fixed_threshold*100:.0f}%) ---")
    print(classification_report(y_true_point, y_pred_fixed, labels=[0, 1], target_names=['Normal', 'Anomaly'], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true_point, y_pred_fixed, labels=[0, 1]))

    if thresholds and 'point_threshold' in thresholds:
        mse_scores = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))
        classification = Preprocessor.classify_anomalies(
            mse_scores,
            point_threshold=thresholds['point_threshold'],
            collective_threshold=thresholds['collective_threshold'],
            collective_window=thresholds.get('collective_window', 12)
        )
        y_pred_calibrated = classification['is_anomaly'].astype(int)
        print("\n--- Sequence-Level Metrics (Calibrated Point + Collective MSE) ---")
        print(classification_report(y_true_seq, y_pred_calibrated, labels=[0, 1], target_names=['Normal', 'Anomaly'], zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_true_seq, y_pred_calibrated, labels=[0, 1]))
        print(f"Identified Point Anomalies: {np.sum(classification['is_point'])}")
        print(f"Identified Collective Anomalies: {np.sum(classification['is_collective'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM Autoencoder on test data with injected anomalies.")
    parser.add_argument("--data", default=None, help="Path to test CSV or SQLite database file")
    parser.add_argument("--no-inject", action="store_true", help="Evaluate raw data without injecting synthetic anomalies")
    args = parser.parse_args()
    evaluate_model(data_path=args.data, inject_anomalies=not args.no_inject)