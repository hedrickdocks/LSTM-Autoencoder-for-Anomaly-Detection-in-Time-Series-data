import os
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
from train_from_db import (
    _split_chronological,
    _subsample_training_sequences,
    augment_normal_transitions,
    build_temporal_sample_weights,
    calibrate_thresholds_from_validation,
)


def train_model():
    DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'generated_energy_data.csv')
    MODEL_PATH = 'lstm_autoencoder.h5'
    SCALER_PATH = 'scaler.pkl'
    THRESHOLDS_PATH = 'anomaly_thresholds.pkl'
    PLOT_PATH = 'training_loss.png'
    HISTORY_PATH = 'training_history.pkl'
    DATA_PLOT_PATH = 'training_data_plot.png'
    TIME_STEPS = 24
    FEATURE_COLUMNS = list(STATIONARY_MODEL_FEATURE_COLUMNS)
    N_FEATURES = len(FEATURE_COLUMNS)

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Initializing preprocessor...")
    preprocessor = Preprocessor()

    print("Loading data...")
    df = preprocessor.load_data(DATA_PATH)
    df_feat_all = preprocessor.add_cyclical_time_features(df, timezone=DATABASE_TIMEZONE)
    raw_power_all = pd.to_numeric(df_feat_all['power'], errors='coerce').fillna(0.0)
    df_clean = preprocessor.clean_data(df_feat_all[FEATURE_COLUMNS])

    df_train_feat, df_val_feat = _split_chronological(df_clean, time_steps=TIME_STEPS)
    train_power = raw_power_all.loc[df_train_feat.index]

    print("Fitting scaler on training split...")
    scaled_train = preprocessor.fit_transform(df_train_feat)
    scaled_val = preprocessor.transform(df_val_feat)
    joblib.dump(preprocessor.scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    print(f"Creating sequences with time_steps={TIME_STEPS} (gap-aware)...")
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
        print("Error: Not enough continuous data for chronological split.")
        return

    stride = 3 if len(X_train) > 60000 else 1
    steady_train, real_trans = _subsample_training_sequences(X_train, FEATURE_COLUMNS, stride)
    rng = np.random.default_rng(42)
    aug_trans = augment_normal_transitions(steady_train, rng, FEATURE_COLUMNS, n_samples=20000)
    parts = [steady_train]
    if len(real_trans) > 0:
        parts.append(np.repeat(real_trans, 10, axis=0))
    if len(aug_trans) > 0:
        parts.append(aug_trans)
    X_train = np.concatenate(parts, axis=0)
    X_train = X_train[rng.permutation(len(X_train))]

    X_train_noisy = X_train + rng.normal(0.0, 0.01, size=X_train.shape)

    model = LSTMAutoencoder(TIME_STEPS, N_FEATURES)
    model.summary()

    print("Starting denoising training with EarlyStopping and LR scheduling...")
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
        epochs=25,
        batch_size=256,
        validation_data=(X_val, X_val, val_weights),
        callbacks=callbacks,
    )

    joblib.dump(history.history, HISTORY_PATH)
    print(f"Training history saved to {HISTORY_PATH}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Calibrating inference-aligned thresholds from held-out validation errors...")
    thresholds = calibrate_thresholds_from_validation(model, X_val, collective_window=12)
    train_raw_for_schedule = pd.DataFrame({'power': train_power.to_numpy()}, index=df_train_feat.index)
    thresholds['feature_columns'] = FEATURE_COLUMNS
    thresholds['timezone'] = str(DATABASE_TIMEZONE)
    thresholds['schedule_profile'] = preprocessor.fit_schedule_profile(
        train_raw_for_schedule,
        slot_minutes=15,
        on_power_threshold=1.0,
        timezone=DATABASE_TIMEZONE,
    )
    joblib.dump(thresholds, THRESHOLDS_PATH)
    print(
        f"Anomaly thresholds saved to {THRESHOLDS_PATH}: "
        f"Point={thresholds['point_threshold']:.5f}, Collective={thresholds['collective_threshold']:.5f}"
    )

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss during Training')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_PATH)
        print(f"Training loss plot saved to {PLOT_PATH}")

        print("Plotting daily average of training data...")
        df_daily_avg = df_clean.resample('D').mean()
        plt.figure(figsize=(12, 8))
        for column in df_daily_avg.columns:
            plt.plot(df_daily_avg.index, df_daily_avg[column], label=column.replace('_', ' ').title())
        plt.title('Average Daily Energy Consumption (Training Data)')
        plt.xlabel('Time in days')
        plt.ylabel('Average Scaled/Raw Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(DATA_PLOT_PATH)
        print(f"Training data plot saved to {DATA_PLOT_PATH}")


if __name__ == "__main__":
    train_model()
