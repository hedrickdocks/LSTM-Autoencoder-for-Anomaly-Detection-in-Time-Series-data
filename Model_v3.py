import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout


class LSTMAutoencoder:
    """Right-sized LSTM Autoencoder with a compact latent bottleneck for anomaly detection."""

    def __init__(self, time_steps, n_features):
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(name="LSTM_Autoencoder")
        model.add(Input(shape=(self.time_steps, self.n_features)))
        model.add(LSTM(64, name="Encoder_LSTM_1"))
        model.add(Dropout(rate=0.10))
        model.add(Dense(24, activation="tanh", name="Latent_Bottleneck"))
        model.add(RepeatVector(self.time_steps, name="Repeat_Vector"))
        model.add(LSTM(64, return_sequences=True, name="Decoder_LSTM_1"))
        model.add(Dropout(rate=0.10))
        model.add(TimeDistributed(Dense(self.n_features), name="Output_Layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def summary(self):
        self.model.summary()

    def train(
        self,
        X_train,
        epochs=25,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        X_target=None,
        sample_weight=None,
    ):
        targets = X_train if X_target is None else X_target
        if validation_data is not None:
            return self.model.fit(
                X_train,
                targets,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=verbose,
                callbacks=callbacks,
            )
        return self.model.fit(
            X_train,
            targets,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict(self, X, batch_size=512, verbose=0):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def save(self, filepath):
        self.model.save(filepath)


if __name__ == "__main__":
    autoencoder = LSTMAutoencoder(time_steps=24, n_features=8)
    autoencoder.summary()
