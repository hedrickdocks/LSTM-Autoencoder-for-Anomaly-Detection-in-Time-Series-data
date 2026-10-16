import os
import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from Preprocessor import MODEL_FEATURE_COLUMNS, Preprocessor
from Model_v3 import LSTMAutoencoder

def train_model():
    DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'generated_energy_data.csv')
    MODEL_PATH = 'lstm_autoencoder.h5'
    SCALER_PATH = 'scaler.pkl'
    THRESHOLDS_PATH = 'anomaly_thresholds.pkl'
    PLOT_PATH = 'training_loss.png'
    HISTORY_PATH = 'training_history.pkl'
    DATA_PLOT_PATH = 'training_data_plot.png'
    TIME_STEPS = 24
    N_FEATURES = len(MODEL_FEATURE_COLUMNS)

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Initializing preprocessor...")
    preprocessor = Preprocessor()

    print("Loading data...")
    df = preprocessor.load_data(DATA_PATH)
    df = preprocessor.add_cyclical_time_features(df)
    df = df[MODEL_FEATURE_COLUMNS]
    df_clean = preprocessor.clean_data(df)

    print("Scaling data...")
    scaled_data = preprocessor.fit_transform(df_clean)
    joblib.dump(preprocessor.scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    print(f"Creating sequences with time_steps={TIME_STEPS} (gap-aware)...")
    sequences, _ = preprocessor.create_sequences_gap_aware(
        scaled_data,
        timestamps=df_clean.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15
    )
    print(f"Data shape: {sequences.shape}")

    model = LSTMAutoencoder(TIME_STEPS, N_FEATURES)
    model.summary()

    print("Starting training...")
    history = model.train(sequences, epochs=30, batch_size=32, validation_split=0.1)

    joblib.dump(history.history, HISTORY_PATH)
    print(f"Training history saved to {HISTORY_PATH}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Calibrating anomaly thresholds from training reconstruction errors...")
    reconstructions = model.predict(sequences)
    train_mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    thresholds = preprocessor.compute_anomaly_thresholds(
        train_mse,
        point_sigma=3.0,
        collective_sigma=1.5,
        collective_window=12
    )
    joblib.dump(thresholds, THRESHOLDS_PATH)
    print(f"Anomaly thresholds saved to {THRESHOLDS_PATH}: Point={thresholds['point_threshold']:.5f}, Collective={thresholds['collective_threshold']:.5f}")

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
            plt.plot(df_daily_avg.index, df_daily_avg[column], label=column.replace('_watts', '').replace('_', ' ').title())
        plt.title('Average Daily Energy Consumption (Training Data)')
        plt.xlabel('Time in days')
        plt.ylabel('Average Daily Wattage')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(DATA_PLOT_PATH)
        print(f"Training data plot saved to {DATA_PLOT_PATH}")

if __name__ == "__main__":
    train_model()
