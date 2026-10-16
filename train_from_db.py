import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from Preprocessor import Preprocessor, MODEL_FEATURE_COLUMNS
from Model_v3 import LSTMAutoencoder

def train_from_db(db_name='test_data.db'):
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', db_name))
    if not os.path.exists(target_path):
        fallback_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'local.db'))
        if os.path.exists(fallback_path):
            db_path = fallback_path
        else:
            print(f"Error: Database not found at {target_path} or {fallback_path}")
            return
    else:
        db_path = target_path

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY datetime ASC", conn)
    conn.close()

    if df.empty:
        print("Error: No data found in energy_data table.")
        return
    print(f"Loaded {len(df)} rows from the database.")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    print("Initializing preprocessor...")
    preprocessor = Preprocessor()

    df = preprocessor.add_cyclical_time_features(df)
    df = df[MODEL_FEATURE_COLUMNS]
    df_clean = preprocessor.clean_data(df)

    print("Scaling data...")
    scaled_data = preprocessor.fit_transform(df_clean)
    joblib.dump(preprocessor.scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")

    TIME_STEPS = 24
    N_FEATURES = len(MODEL_FEATURE_COLUMNS)

    print(f"Creating sequences with time_steps={TIME_STEPS} (gap-aware)...")
    sequences, _ = preprocessor.create_sequences_gap_aware(
        scaled_data,
        timestamps=df_clean.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15
    )

    if len(sequences) == 0:
        print(f"Error: Not enough data to create sequences. At least {TIME_STEPS} continuous points are required.")
        return
    print(f"Sequences shape: {sequences.shape}")

    model = LSTMAutoencoder(TIME_STEPS, N_FEATURES)
    model.summary()

    print("Starting training...")
    history = model.train(sequences, epochs=30, batch_size=32, validation_split=0.1)

    joblib.dump(history.history, 'training_history.pkl')
    print("Training history saved to training_history.pkl")

    model.save('lstm_autoencoder.h5')
    print("Model saved to lstm_autoencoder.h5")

    print("Calibrating anomaly thresholds from training reconstruction errors...")
    reconstructions = model.predict(sequences)
    train_mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    thresholds = preprocessor.compute_anomaly_thresholds(
        train_mse,
        point_sigma=3.0,
        collective_sigma=1.5,
        collective_window=12
    )
    joblib.dump(thresholds, 'anomaly_thresholds.pkl')
    print(f"Anomaly thresholds saved to anomaly_thresholds.pkl: Point={thresholds['point_threshold']:.5f}, Collective={thresholds['collective_threshold']:.5f}")

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
        plot_path = 'training_loss_db.png'
        plt.savefig(plot_path)
        print(f"Training loss plot saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder from SQLite database.")
    parser.add_argument("--db", default="test_data.db", help="Database file name in backend directory (default: test_data.db)")
    args = parser.parse_args()
    train_from_db(db_name=args.db)
