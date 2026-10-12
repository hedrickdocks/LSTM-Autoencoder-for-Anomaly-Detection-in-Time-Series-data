import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, auc
from keras.losses import MeanSquaredError
import os

from Preprocessor import MODEL_FEATURE_COLUMNS, Preprocessor

MODEL_PATH = 'lstm_autoencoder.h5'
SCALER_PATH = 'scaler.pkl'
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'generated_energy_data.csv')
TIME_STEPS = 24

def get_labeled_test_data():
    """
    Loads real test data and injects known anomalies for evaluation.
    Returns:
        df (pd.DataFrame): Data features.
        labels (np.array): Binary labels (0=Normal, 1=Anomaly) aligned with timestamps.
    """

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"{TEST_DATA_PATH} not found.")

    df = pd.read_csv(TEST_DATA_PATH, parse_dates=True, index_col=0, on_bad_lines='skip', engine='python')
    df = df[MODEL_FEATURE_COLUMNS[:5]]

    n_samples = len(df)
    if n_samples < 150:
        raise ValueError(f"Test data is too short ({n_samples} rows). Please provide a CSV with at least 150 rows to support anomaly injection.")
    labels = np.zeros(n_samples, dtype=int)
    spike_indices = [30, 80, 140]
    for idx in spike_indices:
        if idx < n_samples:
            df.iloc[idx, df.columns.get_loc('power')] = 15000
            labels[idx] = 1

    return df, labels

def evaluate_model():
    print("--- Starting Evaluation ---")

    try:
        model = tf.keras.models.load_model(MODEL_PATH,
                                           custom_objects={ "mse": MeanSquaredError() })
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return

    df, raw_labels = get_labeled_test_data()

    prep = Preprocessor()
    prep.scaler = scaler
    prep.is_fitted = True

    THRESHOLDS_PATH = 'anomaly_thresholds.pkl'
    thresholds = None
    if os.path.exists(THRESHOLDS_PATH):
        try:
            thresholds = joblib.load(THRESHOLDS_PATH)
            print(f"Loaded calibrated thresholds: Point={thresholds.get('point_threshold'):.4f}, Collective={thresholds.get('collective_threshold'):.4f}")
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
    y_true = np.array([timestamp_to_label.get(ts, 0) for ts in target_timestamps])

    print(f"Test Data Shape: {X_test.shape}")
    print(f"Anomaly Count: {np.sum(y_true)} / {len(y_true)}")

    print("Running inference...")
    X_pred = model.predict(X_test, verbose=0)

    original_last = X_test[:, -1, :]
    recon_last = X_pred[:, -1, :]

    orig_rescaled = prep.inverse_transform(original_last)
    recon_rescaled = prep.inverse_transform(recon_last)

    baseline = np.maximum(recon_rescaled, 1.0)
    diff = np.abs(orig_rescaled - recon_rescaled)
    deviation_ratio = diff / baseline
    anomaly_scores = np.max(deviation_ratio, axis=1)
    fixed_threshold = 0.50
    y_pred_fixed = (anomaly_scores > fixed_threshold).astype(int)

    print(f"\n--- Metrics for Fixed Threshold > {fixed_threshold*100}% ---")
    print(classification_report(y_true, y_pred_fixed, target_names=['Normal', 'Anomaly'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_fixed))

    if thresholds and 'point_threshold' in thresholds:
        mse_scores = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))
        classification = Preprocessor.classify_anomalies(
            mse_scores,
            point_threshold=thresholds['point_threshold'],
            collective_threshold=thresholds['collective_threshold'],
            collective_window=thresholds.get('collective_window', 12)
        )
        y_pred_calibrated = classification['is_anomaly'].astype(int)
        print("\n--- Metrics for Calibrated Thresholds (Point + Collective) ---")
        print(classification_report(y_true, y_pred_calibrated, target_names=['Normal', 'Anomaly'], zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred_calibrated))
        print(f"Identified Point Anomalies: {np.sum(classification['is_point'])}")
        print(f"Identified Collective Anomalies: {np.sum(classification['is_collective'])}")

if __name__ == "__main__":
    evaluate_model()
