import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import MeanSquaredError

try:
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from Preprocessor import MODEL_FEATURE_COLUMNS, Preprocessor

MODEL_PATH = 'lstm_autoencoder.h5'
SCALER_PATH = 'scaler.pkl'
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'generated_energy_data.csv')
TIME_STEPS = 24

def run_inference():
    print("--- Starting Inference Pipeline ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading files: {e}. Make sure model and scaler exist.")
        return

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    df_test = pd.read_csv(TEST_DATA_PATH)
    df_test['datetime'] = pd.to_datetime(df_test['datetime'], utc=True)
    df_test = df_test.head(168)

    THRESHOLDS_PATH = 'anomaly_thresholds.pkl'
    thresholds = None
    if os.path.exists(THRESHOLDS_PATH):
        try:
            thresholds = joblib.load(THRESHOLDS_PATH)
            print(f"Loaded calibrated thresholds: Point={thresholds.get('point_threshold'):.4f}, Collective={thresholds.get('collective_threshold'):.4f}")
        except Exception as e:
            print(f"Could not load thresholds: {e}")

    prep = Preprocessor()
    prep.scaler = scaler
    prep.is_fitted = True

    df_test = prep.add_cyclical_time_features(df_test)
    df_features = df_test[MODEL_FEATURE_COLUMNS]
    scaled_test = prep.transform(df_features)
    X_test, target_timestamps = prep.create_sequences_gap_aware(
        scaled_test,
        timestamps=df_test.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15
    )
    print(f"Test sequences shape: {X_test.shape}")

    if len(X_test) == 0:
        print("No valid continuous sequences found.")
        return

    print("Running model inference...")
    X_pred = model.predict(X_test, verbose=0)
    mse_scores = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))

    point_thresh = thresholds.get('point_threshold', 0.50) if thresholds else 0.50
    collective_thresh = thresholds.get('collective_threshold', 0.30) if thresholds else 0.30
    collective_win = thresholds.get('collective_window', 12) if thresholds else 12

    classification = Preprocessor.classify_anomalies(
        mse_scores,
        point_threshold=point_thresh,
        collective_threshold=collective_thresh,
        collective_window=collective_win
    )

    print(f"\nDetection Results:")
    print(f"Total Sequences: {len(mse_scores)}")
    print(f"Point Anomalies: {np.sum(classification['is_point'])} (Threshold: > {point_thresh:.4f})")
    print(f"Collective Anomalies: {np.sum(classification['is_collective'])} (Threshold: > {collective_thresh:.4f} sustained over {collective_win} steps)")
    print(f"Total Anomalous Timestamps: {np.sum(classification['is_anomaly'])}")

    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not found. Skipping plot generation.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(df_features.columns[:5]):
        ax1.plot(df_test.index, df_test[col], label=col.replace('_', ' ').title(), color=colors[i % len(colors)], alpha=0.5, linewidth=1)

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Energy Consumption (Watts)')
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    target_dates = pd.to_datetime(target_timestamps)
    ax2.plot(target_dates, mse_scores, label='Reconstruction Error (MSE)', color='black', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel('Reconstruction Error (MSE)')

    is_anomaly = classification['is_anomaly']
    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        ax2.scatter(target_dates[anomaly_indices], mse_scores[anomaly_indices], color='red', label='Detected Anomaly', zorder=5, s=50)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Anomaly Detection on Test Data: Energy vs Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('inference_results.png')
    print("Inference plot saved to 'inference_results.png'")

if __name__ == "__main__":
    run_inference()
