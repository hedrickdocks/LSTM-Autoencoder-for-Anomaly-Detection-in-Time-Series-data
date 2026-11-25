import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from Preprocessor import (
    DATABASE_TIMEZONE,
    Preprocessor,
    STATIONARY_MODEL_FEATURE_COLUMNS,
)
from Model_v3 import LSTMAutoencoder


def _split_chronological(df, time_steps=24):
    """Leak-free chronological train/validation split on raw telemetry rows."""
    df_sorted = df.sort_index()
    span_days = (df_sorted.index.max() - df_sorted.index.min()).total_seconds() / 86400.0
    if span_days >= 10.0:
        cutoff = df_sorted.index.max() - pd.Timedelta(days=7)
        df_train = df_sorted[df_sorted.index <= cutoff]
        df_val = df_sorted[df_sorted.index > cutoff]
        if not df_train.empty and len(df_val) > time_steps:
            return df_train, df_val

    split_idx = int(len(df_sorted) * 0.8)
    embargo_end = min(len(df_sorted), split_idx + time_steps)
    return df_sorted.iloc[:split_idx], df_sorted.iloc[embargo_end:]


def _subsample_training_sequences(X_train, feature_columns, stride):
    """Subsample steady training windows by `stride` while isolating 100% of real transition windows."""
    power_col = feature_columns.index('power')
    is_trans = np.ptp(X_train[:, :, power_col], axis=1) > 0.03
    steady = X_train[~is_trans]
    real_trans = X_train[is_trans]
    if stride > 1 and len(steady) > 10000:
        steady = steady[::stride]
    return steady, real_trans


def build_temporal_sample_weights(n_samples, time_steps=24, last_step_weight=5.0, prev_step_weight=2.0):
    """Weight final timesteps higher so last-step point MSE reconstructs switch edges accurately
    while keeping >80% of total weight on sequence context (steps 0..22)."""
    w = np.ones((n_samples, time_steps), dtype=np.float32)
    if time_steps >= 2:
        w[:, -2] = prev_step_weight
    w[:, -1] = last_step_weight
    w /= float(w[0].mean())
    return w


def augment_normal_transitions(X_steady, rng, feature_columns, n_samples=20000):
    """Synthesize normal 1-step and 2-step switching windows (with PZEM mid-window blends)
    from steady training templates so step transitions reconstruct as accurately as steady states."""
    if len(X_steady) == 0 or n_samples <= 0:
        return np.empty((0,) + X_steady.shape[1:], dtype=X_steady.dtype)

    curr_col = feature_columns.index('current')
    pow_col = feature_columns.index('power')
    pf_col = feature_columns.index('power_factor')
    elec_cols = [curr_col, pow_col, pf_col]
    time_steps = X_steady.shape[1]

    last_elec = X_steady[:, -1, :][:, elec_cols]
    off_pool = last_elec[last_elec[:, 1] < 0.01]
    on_base = last_elec[last_elec[:, 1] >= 0.05]

    if len(off_pool) == 0 or len(on_base) == 0:
        return np.empty((0,) + X_steady.shape[1:], dtype=X_steady.dtype)

    # Preserve full density of observed ON levels (e.g., 1..4 fans Stg 1..3),
    # adding only a small fraction of scaled intermediate levels above min observed ON power.
    min_on_power = max(0.15, float(np.percentile(on_base[:, 1], 5)) * 0.5)
    scaled_parts = []
    for scale in (0.50, 0.67, 0.75):
        lvl = on_base.copy()
        lvl[:, 0] *= scale
        lvl[:, 1] *= scale
        lvl = lvl[lvl[:, 1] >= min_on_power]
        if len(lvl) > 0:
            scaled_parts.append(lvl)
    scaled_pool = np.vstack(scaled_parts) if scaled_parts else on_base

    def pick_off(n):
        return off_pool[rng.integers(0, len(off_pool), size=n)]

    def pick_on(n):
        use_raw = rng.random(n) < 0.80
        out = np.empty((n, 3), dtype=on_base.dtype)
        n_raw = int(np.sum(use_raw))
        if n_raw > 0:
            out[use_raw] = on_base[rng.integers(0, len(on_base), size=n_raw)]
        if n - n_raw > 0:
            out[~use_raw] = scaled_pool[rng.integers(0, len(scaled_pool), size=n - n_raw)]
        return out

    tmpl_idx = rng.integers(0, len(X_steady), size=n_samples)
    aug = X_steady[tmpl_idx].copy()
    aug_elec = aug[:, :, elec_cols].copy()
    n_single = int(n_samples * 0.7)

    # 70% single-step transitions (60% OFF<->ON, 40% level<->level), biased toward late switch edges
    late_cuts = np.array([time_steps - 4, time_steps - 3, time_steps - 2, time_steps - 1])
    late_probs = np.array([0.15, 0.20, 0.25, 0.40])
    for i in range(n_single):
        r = rng.random()
        if r < 0.30:
            s_a = pick_off(1)[0]
            s_b = pick_on(1)[0]
        elif r < 0.60:
            s_a = pick_on(1)[0]
            s_b = pick_off(1)[0]
        else:
            s_a = pick_on(1)[0]
            s_b = pick_on(1)[0]
        if rng.random() < 0.45 and time_steps >= 5:
            t_cut = int(rng.choice(late_cuts, p=late_probs))
        else:
            t_cut = int(rng.integers(1, max(2, time_steps - 4)))
        aug_elec[i, :t_cut] = s_a
        aug_elec[i, t_cut:] = s_b
        if rng.random() < 0.5:
            alpha = float(rng.uniform(0.06, 0.94))
            aug_elec[i, t_cut] = alpha * s_a + (1.0 - alpha) * s_b

    # 30% two-step transitions (e.g. OFF -> 1-bulb -> 3-bulb, or 3-bulb -> 1-bulb -> OFF)
    cuts_all = np.arange(1, time_steps)
    late_t2 = np.array([time_steps - 3, time_steps - 2, time_steps - 1])
    late_t2_probs = np.array([0.20, 0.30, 0.50])
    for i in range(n_single, n_samples):
        r = rng.random()
        if r < 0.45:
            s_a = pick_off(1)[0]
            s_mid = pick_on(1)[0]
            s_b = pick_on(1)[0]
        elif r < 0.85:
            s_a = pick_on(1)[0]
            s_mid = pick_on(1)[0]
            s_b = pick_off(1)[0]
        else:
            s_a = pick_off(1)[0]
            s_mid = pick_on(1)[0]
            s_b = pick_off(1)[0]
        if rng.random() < 0.40 and time_steps >= 5:
            t2 = int(rng.choice(late_t2, p=late_t2_probs))
            t1 = int(rng.integers(1, t2))
        else:
            t1, t2 = sorted(rng.choice(cuts_all, size=2, replace=False).tolist())
        aug_elec[i, :t1] = s_a
        aug_elec[i, t1:t2] = s_mid
        aug_elec[i, t2:] = s_b
        if rng.random() < 0.5:
            alpha = float(rng.uniform(0.06, 0.94))
            aug_elec[i, t1] = alpha * s_a + (1.0 - alpha) * s_mid
        if rng.random() < 0.5:
            alpha = float(rng.uniform(0.06, 0.94))
            aug_elec[i, t2] = alpha * s_mid + (1.0 - alpha) * s_b

    aug[:, :, elec_cols] = aug_elec
    return aug


def calibrate_thresholds_from_validation(model, X_val, collective_window=12):
    """Calibrate thresholds using inference-aligned last-step point MSE and rolling average."""
    val_recon = model.predict(X_val, batch_size=512, verbose=0)
    val_point_mse = np.mean(np.power(X_val[:, -1, :] - val_recon[:, -1, :], 2), axis=1)
    val_rolling = (
        pd.Series(val_point_mse)
        .rolling(window=collective_window, min_periods=collective_window)
        .mean()
        .dropna()
        .to_numpy()
    )
    if len(val_rolling) == 0:
        val_rolling = val_point_mse

    p99_point = float(np.percentile(val_point_mse, 99))
    p999_point = float(np.percentile(val_point_mse, 99.9))
    p9999_point = float(np.percentile(val_point_mse, 99.99))
    p995_rolling = float(np.percentile(val_rolling, 99.5))
    p999_rolling = float(np.percentile(val_rolling, 99.9))

    point_threshold = max(p9999_point * 1.5, 0.04)
    collective_threshold = min(max(p999_rolling * 1.4, 0.012), 0.35 * point_threshold)
    schedule_penalty_scale = min(collective_threshold * 2.0, point_threshold * 0.75)

    return {
        'point_threshold': float(point_threshold),
        'collective_threshold': float(collective_threshold),
        'collective_window': int(collective_window),
        'schedule_penalty_scale': float(schedule_penalty_scale),
        'mean_error': float(np.mean(val_point_mse)),
        'std_error': float(np.std(val_point_mse)),
        'p99_point': p99_point,
        'p999_point': p999_point,
        'p9999_point': p9999_point,
        'p995_rolling': p995_rolling,
        'p999_rolling': p999_rolling,
    }


def train_from_db(
    db_name='test_data.db',
    load=None,
    epochs=25,
    batch_size=256,
    train_stride=None,
    recalibrate_only=False,
):
    """Trains the default model on room totals (energy_data), or, with load=..., a model for one
    load (lights, fan, outlet, aircon) on its own readings (load_readings), saved to models/<load>/."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'models', load) if load else base_dir
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, name)

    target_path = os.path.abspath(os.path.join(base_dir, '..', 'backend', db_name))
    if not os.path.exists(target_path):
        fallback_path = os.path.abspath(os.path.join(base_dir, '..', 'backend', 'local.db'))
        if os.path.exists(fallback_path):
            db_path = fallback_path
        else:
            print(f"Error: Database not found at {target_path} or {fallback_path}")
            return
    else:
        db_path = target_path

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    if load:
        df = pd.read_sql_query("SELECT * FROM load_readings WHERE load = ? ORDER BY datetime ASC", conn, params=(load,))
    else:
        df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY datetime ASC", conn)
    conn.close()

    if df.empty:
        print(f"Error: No data found for {'load ' + load if load else 'energy_data'}.")
        return
    print(f"Loaded {len(df)} rows from the database.")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    feature_columns = list(STATIONARY_MODEL_FEATURE_COLUMNS)
    TIME_STEPS = 24
    N_FEATURES = len(feature_columns)

    print("Initializing preprocessor (Asia/Manila cyclical time + 8 stationary features)...")
    preprocessor = Preprocessor()
    df_feat_all = preprocessor.add_cyclical_time_features(df, timezone=DATABASE_TIMEZONE)
    raw_power_all = pd.to_numeric(df_feat_all['power'], errors='coerce').fillna(0.0)
    df_feat_all = preprocessor.clean_data(df_feat_all[feature_columns])

    df_train_feat, df_val_feat = _split_chronological(df_feat_all, time_steps=TIME_STEPS)
    train_power = raw_power_all.loc[df_train_feat.index]
    print(f"Chronological split -> train rows: {len(df_train_feat)}, val rows: {len(df_val_feat)}")

    if recalibrate_only:
        print("Recalibrate-only mode: loading existing model and scaler...")
        preprocessor.scaler = joblib.load(out('scaler.pkl'))
        preprocessor.is_fitted = True
        scaled_val = preprocessor.transform(df_val_feat)
        X_val, _ = preprocessor.create_sequences_gap_aware(
            scaled_val,
            timestamps=df_val_feat.index,
            time_steps=TIME_STEPS,
            max_gap_seconds=15,
            energy_col_idx=None,
        )
        if len(X_val) == 0:
            print("Error: Not enough continuous validation data for recalibration.")
            return
        model = tf.keras.models.load_model(
            out('lstm_autoencoder.h5'),
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()},
        )
        history = None
    else:
        print("Fitting scaler on training split only...")
        scaled_train = preprocessor.fit_transform(df_train_feat)
        scaled_val = preprocessor.transform(df_val_feat)
        joblib.dump(preprocessor.scaler, out('scaler.pkl'))
        print(f"Scaler saved to {out('scaler.pkl')}")

        print(f"Creating gap-aware sequences (time_steps={TIME_STEPS}, energy_col_idx=None)...")
        X_train, _ = preprocessor.create_sequences_gap_aware(
            scaled_train,
            timestamps=df_train_feat.index,
            time_steps=TIME_STEPS,
            max_gap_seconds=15,
            energy_col_idx=None,
        )
        X_val, _ = preprocessor.create_sequences_gap_aware(
            scaled_val,
            timestamps=df_val_feat.index,
            time_steps=TIME_STEPS,
            max_gap_seconds=15,
            energy_col_idx=None,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Error: Not enough continuous data to form sequences (train={len(X_train)}, val={len(X_val)}).")
            return

        effective_stride = train_stride if train_stride is not None else (3 if len(X_train) > 60000 else 1)
        steady_train, real_trans = _subsample_training_sequences(X_train, feature_columns, effective_stride)

        rng = np.random.default_rng(42)
        aug_trans = augment_normal_transitions(steady_train, rng, feature_columns, n_samples=20000)
        parts = [steady_train]
        if len(real_trans) > 0:
            repeat_factor = min(10, max(1, 25000 // len(real_trans)))
            parts.append(np.repeat(real_trans, repeat_factor, axis=0))
        else:
            repeat_factor = 0
        if len(aug_trans) > 0:
            parts.append(aug_trans)
        X_train = np.concatenate(parts, axis=0)
        X_train = X_train[rng.permutation(len(X_train))]
        print(
            f"Train sequences: {X_train.shape} "
            f"(steady={len(steady_train)}, real_trans={len(real_trans)}x{repeat_factor}, aug_trans={len(aug_trans)}), "
            f"Validation sequences: {X_val.shape}"
        )

        X_train_noisy = X_train + rng.normal(0.0, 0.01, size=X_train.shape)

        model = LSTMAutoencoder(TIME_STEPS, N_FEATURES)
        model.summary()

        print("Starting denoising autoencoder training...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ]
        train_weights = build_temporal_sample_weights(len(X_train), TIME_STEPS)
        val_weights = build_temporal_sample_weights(len(X_val), TIME_STEPS)
        history = model.train(
            X_train_noisy,
            X_target=X_train,
            sample_weight=train_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val, val_weights),
            callbacks=callbacks,
        )

        joblib.dump(history.history, out('training_history.pkl'))
        print(f"Training history saved to {out('training_history.pkl')}")

        model.save(out('lstm_autoencoder.h5'))
        print(f"Model saved to {out('lstm_autoencoder.h5')}")

    print("Calibrating inference-aligned thresholds and temporal schedule profile...")
    thresholds = calibrate_thresholds_from_validation(model, X_val, collective_window=12)
    train_raw_for_schedule = pd.DataFrame({'power': train_power.to_numpy()}, index=df_train_feat.index)
    schedule_profile = preprocessor.fit_schedule_profile(
        train_raw_for_schedule,
        slot_minutes=15,
        on_power_threshold=1.0,
        timezone=DATABASE_TIMEZONE,
        per_slot_power_cap=(load != 'fan'),
    )
    thresholds['feature_columns'] = feature_columns
    thresholds['timezone'] = str(DATABASE_TIMEZONE)
    thresholds['schedule_profile'] = schedule_profile

    joblib.dump(thresholds, out('anomaly_thresholds.pkl'))
    print(
        f"Anomaly thresholds saved to {out('anomaly_thresholds.pkl')}: "
        f"Point={thresholds['point_threshold']:.5f}, "
        f"Collective={thresholds['collective_threshold']:.5f}, "
        f"SchedulePenalty={thresholds['schedule_penalty_scale']:.5f}"
    )

    if HAS_MATPLOTLIB and history is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss during Training')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plot_path = out('training_loss_db.png')
        plt.savefig(plot_path)
        print(f"Training loss plot saved to {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder from SQLite database.")
    parser.add_argument("--db", default="test_data.db", help="Database file name in backend directory (default: test_data.db)")
    parser.add_argument("--load", choices=["lights", "fan", "outlet", "aircon"], help="Train a model for one load (saved to models/<load>/); omit for room totals")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--recalibrate-only", action="store_true", help="Recalibrate thresholds and schedule profile using existing model/scaler")
    args = parser.parse_args()
    train_from_db(
        db_name=args.db,
        load=args.load,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        recalibrate_only=args.recalibrate_only,
    )
