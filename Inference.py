import os
import argparse
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

from Preprocessor import (
    DATABASE_TIMEZONE,
    MODEL_FEATURE_COLUMNS,
    Preprocessor,
    STATIONARY_MODEL_FEATURE_COLUMNS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_STEPS = 24


def resolve_artifact_paths(load=None):
    if load:
        folder = os.path.join(BASE_DIR, 'models', load)
        return (
            os.path.join(folder, 'lstm_autoencoder.h5'),
            os.path.join(folder, 'scaler.pkl'),
            os.path.join(folder, 'anomaly_thresholds.pkl'),
        )
    return (
        os.path.join(BASE_DIR, 'lstm_autoencoder.h5'),
        os.path.join(BASE_DIR, 'scaler.pkl'),
        os.path.join(BASE_DIR, 'anomaly_thresholds.pkl'),
    )


def run_inference(data_path=None, load=None, limit=500, output_dir=None):
    print(f"--- Starting Inference Pipeline (load={load or 'default'}) ---")
    model_path, scaler_path, thresholds_path = resolve_artifact_paths(load)
    test_data_path = data_path or os.getenv('TEST_DATA_PATH', os.path.join(BASE_DIR, 'generated_energy_data.csv'))

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading files: {e}. Make sure model and scaler exist.")
        return

    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found.")
        return

    df_test = pd.read_csv(test_data_path)
    df_test['datetime'] = pd.to_datetime(df_test['datetime'], utc=True)
    if limit and limit > 0:
        df_test = df_test.head(limit)

    thresholds = None
    if os.path.exists(thresholds_path):
        try:
            thresholds = joblib.load(thresholds_path)
            print(
                f"Loaded calibrated thresholds: Point={thresholds.get('point_threshold'):.4f}, "
                f"Collective={thresholds.get('collective_threshold'):.4f}"
            )
        except Exception as e:
            print(f"Could not load thresholds: {e}")

    feature_columns = thresholds.get('feature_columns') if thresholds else None
    if not feature_columns:
        n_in = getattr(scaler, 'n_features_in_', 9)
        feature_columns = list(MODEL_FEATURE_COLUMNS if n_in == 9 else STATIONARY_MODEL_FEATURE_COLUMNS)
    tz_name = (thresholds.get('timezone') if thresholds else None) or (
        'UTC' if len(feature_columns) == 9 else str(DATABASE_TIMEZONE)
    )
    energy_col_idx = feature_columns.index('energy') if 'energy' in feature_columns else None

    prep = Preprocessor()
    prep.scaler = scaler
    prep.is_fitted = True

    df_with_time = prep.add_cyclical_time_features(df_test, timezone=tz_name)
    raw_power_by_ts = (
        pd.Series(pd.to_numeric(df_with_time['power'], errors='coerce').fillna(0.0).to_numpy(dtype=float), index=df_with_time.index)
        .groupby(level=0)
        .last()
    )
    df_features = df_with_time[feature_columns].fillna(0.0)
    scaled_test = prep.transform(df_features)
    X_test, target_timestamps = prep.create_sequences_gap_aware(
        scaled_test,
        timestamps=df_with_time.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15,
        energy_col_idx=energy_col_idx,
    )
    print(f"Test sequences shape: {X_test.shape}")

    if len(X_test) == 0:
        print("No valid continuous sequences found.")
        return

    print("Running model inference...")
    X_pred = model.predict(X_test, batch_size=512, verbose=0)
    point_mse = np.mean(np.power(X_test[:, -1, :] - X_pred[:, -1, :], 2), axis=1)

    point_thresh = thresholds.get('point_threshold', 0.50) if thresholds else 0.50
    collective_thresh = thresholds.get('collective_threshold', 0.30) if thresholds else 0.30
    collective_win = thresholds.get('collective_window', 12) if thresholds else 12

    is_point = point_mse > point_thresh
    schedule_profile = thresholds.get('schedule_profile') if thresholds else None
    if schedule_profile:
        target_idx = pd.DatetimeIndex(target_timestamps)
        target_powers = raw_power_by_ts.reindex(target_idx).fillna(0.0).to_numpy()
        penalty_scale = float(thresholds.get('schedule_penalty_scale', collective_thresh * 2.0))
        schedule_penalty = Preprocessor.compute_schedule_penalty(
            target_idx,
            target_powers,
            schedule_profile,
            penalty_scale,
        )
        collective_signal = np.maximum(point_mse, schedule_penalty)
    else:
        collective_signal = point_mse

    clean_for_rolling = np.where(is_point, 0.0, collective_signal)
    rolling_err = (
        pd.Series(clean_for_rolling)
        .rolling(window=collective_win, min_periods=min(collective_win, 4))
        .mean()
        .fillna(0.0)
        .to_numpy()
    )
    is_collective = (rolling_err > collective_thresh) & (~is_point)
    is_anomaly = is_point | is_collective

    summary_lines = [
        f"--- Inference Summary (load={load or 'default'}) ---",
        f"Input File: {test_data_path}",
        f"Input Rows: {len(df_test)}",
        f"Total Sequences Scored: {len(point_mse)}",
        f"Point Threshold: {point_thresh:.5f}",
        f"Collective Threshold: {collective_thresh:.5f} (window={collective_win})",
        f"Point MSE -> min={float(np.min(point_mse)):.6f}, median={float(np.median(point_mse)):.6f}, p99={float(np.percentile(point_mse, 99)):.6f}, max={float(np.max(point_mse)):.6f}",
        f"Rolling Error -> min={float(np.min(rolling_err)):.6f}, median={float(np.median(rolling_err)):.6f}, max={float(np.max(rolling_err)):.6f}",
        f"Point Anomalies: {int(np.sum(is_point))}",
        f"Collective Anomalies: {int(np.sum(is_collective))}",
        f"Total Anomalous Timestamps: {int(np.sum(is_anomaly))}",
    ]
    print("\nDetection Results:")
    for line in summary_lines[1:]:
        print(line)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, 'inference_output.txt')
        csv_path = os.path.join(output_dir, 'inference_predictions.csv')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines) + '\n')

        target_idx = pd.DatetimeIndex(target_timestamps)
        raw_lookup = (
            df_with_time[['voltage', 'current', 'power', 'power_factor']]
            .groupby(level=0)
            .last()
            .reindex(target_idx)
        )
        anomaly_types = np.where(is_point, 'point', np.where(is_collective, 'collective', 'normal'))
        out_df = pd.DataFrame({
            'datetime': [pd.Timestamp(t).tz_convert('UTC').isoformat().replace('+00:00', 'Z') for t in target_idx],
            'manila_time': [pd.Timestamp(t).tz_convert(DATABASE_TIMEZONE).isoformat() for t in target_idx],
            'voltage': raw_lookup['voltage'].to_numpy(),
            'current': raw_lookup['current'].to_numpy(),
            'power': raw_lookup['power'].to_numpy(),
            'power_factor': raw_lookup['power_factor'].to_numpy(),
            'reconstruction_error': point_mse,
            'rolling_error': rolling_err,
            'anomaly_type': anomaly_types,
            'is_point_anomaly': is_point.astype(bool),
            'is_collective_anomaly': is_collective.astype(bool),
            'is_anomaly': is_anomaly.astype(bool),
        })
        out_df.to_csv(csv_path, index=False)
        print(f"Saved inference summary -> {txt_path}")
        print(f"Saved inference predictions CSV -> {csv_path}")

    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not found. Skipping plot generation.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df_with_time.index, df_with_time['power'], label='Power (W)', color='#1f77b4', alpha=0.6, linewidth=1.2)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Power (Watts)')
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    target_dates = pd.to_datetime(target_timestamps)
    ax2.plot(target_dates, point_mse, label='Point Reconstruction Error (MSE)', color='black', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel('Reconstruction Error (MSE)')

    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        ax2.scatter(target_dates[anomaly_indices], point_mse[anomaly_indices], color='red', label='Detected Anomaly', zorder=5, s=50)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Anomaly Detection on Test Data: Power vs Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('inference_results.png')
    print("Inference plot saved to 'inference_results.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on CSV telemetry.")
    parser.add_argument("--data", default=None, help="Path to CSV telemetry file")
    parser.add_argument("--load", choices=["lights", "fan", "outlet", "aircon"], default=None)
    parser.add_argument("--limit", type=int, default=500, help="Max input rows (0 = all rows)")
    parser.add_argument("--output-dir", default=None, help="Directory to save inference_output.txt and inference_predictions.csv")
    args = parser.parse_args()
    run_inference(data_path=args.data, load=args.load, limit=args.limit, output_dir=args.output_dir)
