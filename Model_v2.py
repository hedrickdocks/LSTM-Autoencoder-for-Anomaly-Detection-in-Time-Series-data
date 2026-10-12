import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout

class LSTMAutoencoder:
    """
    LSTM Autoencoder for Anomaly Detection.
    Architecture designed to compress and reconstruct time-series sequences.
    """
    def __init__(self, time_steps, n_features):
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(name="LSTM_Autoencoder")
        model.add(Input(shape=(self.time_steps, self.n_features)))
        model.add(LSTM(128, activation='relu', return_sequences=True, name="Encoder_LSTM_1"))
        model.add(LSTM(64, activation='relu', return_sequences=False, name="Encoder_LSTM_2"))
        model.add(RepeatVector(self.time_steps, name="Repeat_Vector"))
        model.add(LSTM(64, activation='relu', return_sequences=True, name="Decoder_LSTM_1"))
        model.add(LSTM(128, activation='relu', return_sequences=True, name="Decoder_LSTM_2"))
        model.add(TimeDistributed(Dense(self.n_features), name="Output_Layer"))

        model.compile(optimizer='adam', loss='mse')
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, epochs=60, batch_size=32, validation_split=0.1, verbose=1):
        """
        Trains the autoencoder.
        The target is the input itself (X_train) for reconstruction.
        """
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

if __name__ == "__main__":
    import numpy as np

    TIME_STEPS = 24
    N_FEATURES = 1 # Lighting

    print("Building LSTM Autoencoder...")
    autoencoder = LSTMAutoencoder(TIME_STEPS, N_FEATURES)
    autoencoder.summary()

    # Dummy data test
#    print("\nTesting training loop with dummy data...")
#    dummy_data = np.random.random((100, TIME_STEPS, N_FEATURES))
#    autoencoder.train(dummy_data, epochs=2, batch_size=10)
#    print("Model built and tested successfully.")
