import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout

class LSTMAutoencoder:
    """LSTM Autoencoder for Anomaly Detection with Dropout."""
    def __init__(self, time_steps, n_features):
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(name="LSTM_Autoencoder")
        model.add(Input(shape=(self.time_steps, self.n_features)))
        model.add(LSTM(128, name="Encoder_LSTM_1"))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(self.time_steps, name="Repeat_Vector"))
        model.add(LSTM(128, return_sequences=True, name="Decoder_LSTM_1"))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(self.n_features), name="Output_Layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1):
        return self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

if __name__ == "__main__":
    autoencoder = LSTMAutoencoder(time_steps=24, n_features=9)
    autoencoder.summary()
