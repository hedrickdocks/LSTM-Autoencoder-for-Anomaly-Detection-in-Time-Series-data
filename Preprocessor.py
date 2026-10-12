import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import as_strided
from zoneinfo import ZoneInfo

DATABASE_TIMEZONE = ZoneInfo('Asia/Manila')

TELEMETRY_FEATURE_COLUMNS = [
    'voltage',
    'current',
    'power',
    'energy',
    'power_factor'
]

CYCLICAL_FEATURE_COLUMNS = [
    'hour_sin',
    'hour_cos',
    'week_sin',
    'week_cos'
]

MODEL_FEATURE_COLUMNS = TELEMETRY_FEATURE_COLUMNS + CYCLICAL_FEATURE_COLUMNS

class Preprocessor:
    """
    Preprocessor for Autoencoder Anomaly Detection.
    Handles data cleaning, normalization, and sequence generation.
    """
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False

    def load_data(self, filepath):
        """
        Loads data from a CSV file.
        Assumes the first column is datetime or index.
        """
        return pd.read_csv(filepath, parse_dates=True, index_col=0)

    def clean_data(self, df):
        """
        Handles missing values using linear interpolation and forward/backward filling.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not df.isnull().values.any():
            return df

        df_clean = df.copy()
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        df_clean = df_clean.ffill().bfill()
        return df_clean

    def add_cyclical_time_features(self, df):
        """Add UTC time-of-day and day-of-week features to indexed telemetry."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        result = df.copy()
        if 'datetime' in result.columns:
            timestamps = pd.DatetimeIndex(pd.to_datetime(result.pop('datetime')))
        else:
            timestamps = pd.DatetimeIndex(pd.to_datetime(result.index))

        if timestamps.isna().any():
            raise ValueError("Telemetry contains invalid datetime values.")

        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(DATABASE_TIMEZONE)
        else:
            timestamps = timestamps.tz_convert('UTC')

        seconds_since_midnight = (
            timestamps.hour * 3600 +
            timestamps.minute * 60 +
            timestamps.second +
            timestamps.microsecond / 1_000_000
        )
        seconds_since_week_start = timestamps.dayofweek * 86400 + seconds_since_midnight

        result.index = timestamps
        result['hour_sin'] = np.sin(2 * np.pi * seconds_since_midnight / 86400)
        result['hour_cos'] = np.cos(2 * np.pi * seconds_since_midnight / 86400)
        result['week_sin'] = np.sin(2 * np.pi * seconds_since_week_start / 604800)
        result['week_cos'] = np.cos(2 * np.pi * seconds_since_week_start / 604800)
        return result

    def fit_transform(self, data):
        """
        Fits the scaler to the data and returns normalized data.
        """
        transformed_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        return transformed_data

    def transform(self, data):
        """
        Normalizes data using the fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """
        Reverses normalization to get original values.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return self.scaler.inverse_transform(data)

    def create_sequences(self, data, time_steps=24):
        """
        Generates sliding window sequences.

        Args:
            data (np.array): Normalized data matrix (samples, features).
            time_steps (int): Length of the input sequence (lookback).

        Returns:
            np.array: 3D array of shape (num_sequences, time_steps, features).
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Optimization: Use stride_tricks for O(1) memory view creation
        data = np.ascontiguousarray(data)
        n_samples, n_features = data.shape

        if n_samples < time_steps:
            return np.empty((0, time_steps, n_features))

        n_sequences = n_samples - time_steps + 1
        shape = (n_sequences, time_steps, n_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])

        return as_strided(data, shape=shape, strides=strides)

    def create_sequences_gap_aware(self, data, timestamps=None, time_steps=24, max_gap_seconds=15):
        """
        Generates sliding window sequences without bridging across time gaps larger than max_gap_seconds.
        Guarantees temporal continuity within each sequence.

        Args:
            data (np.ndarray or pd.DataFrame): Normalized feature matrix (samples, features) or DataFrame.
            timestamps (pd.DatetimeIndex or pd.Series or iterable, optional): Timestamps corresponding to rows.
                If data is a pd.DataFrame with DatetimeIndex or 'datetime' column, timestamps can be omitted.
            time_steps (int): Length of the input sequence (lookback).
            max_gap_seconds (float): Max allowed gap between consecutive samples (default: 15s for 5s sampling).

        Returns:
            tuple: (sequences_array, target_timestamps_list)
                - sequences_array: np.ndarray of shape (num_valid_sequences, time_steps, n_features).
                - target_timestamps_list: list of timestamps at the final timestep of each sequence.
        """
        if isinstance(data, pd.DataFrame):
            if timestamps is None:
                if isinstance(data.index, pd.DatetimeIndex):
                    timestamps = data.index
                elif 'datetime' in data.columns:
                    timestamps = pd.to_datetime(data['datetime'])
                    data = data.drop(columns=['datetime'])
            matrix = data.values
        elif isinstance(data, np.ndarray):
            matrix = data
        else:
            raise ValueError("data must be a pandas DataFrame or numpy ndarray.")

        if timestamps is None:
            raise ValueError("timestamps must be provided when data does not have DatetimeIndex or 'datetime' column.")

        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(pd.to_datetime(timestamps))

        if len(matrix) < time_steps:
            n_features = matrix.shape[1] if matrix.ndim > 1 else 1
            return np.empty((0, time_steps, n_features)), []

        time_diffs = timestamps.diff().dt.total_seconds().fillna(0)

        is_break = (time_diffs > max_gap_seconds) | (time_diffs < 0)
        segment_ids = is_break.cumsum()

        valid_sequences = []
        target_timestamps = []

        for _, group_indices in pd.Series(range(len(matrix))).groupby(segment_ids):
            indices = group_indices.values
            if len(indices) >= time_steps:
                block_data = matrix[indices]
                block_seqs = self.create_sequences(block_data, time_steps=time_steps)
                if len(block_seqs) > 0:
                    valid_sequences.append(block_seqs)
                    target_timestamps.extend(timestamps.iloc[indices[time_steps - 1:]].tolist())

        if not valid_sequences:
            n_features = matrix.shape[1] if matrix.ndim > 1 else 1
            return np.empty((0, time_steps, n_features)), []

        return np.concatenate(valid_sequences, axis=0), target_timestamps

    @staticmethod
    def compute_anomaly_thresholds(errors, point_sigma=3.0, collective_sigma=1.5, collective_window=12):
        """
        Computes data-driven anomaly detection thresholds from baseline reconstruction errors.

        Args:
            errors (np.ndarray or list): 1D array of reconstruction errors on normal baseline data.
            point_sigma (float): Multiplier of standard deviation for Point Anomaly threshold (default: 3.0).
            collective_sigma (float): Multiplier of standard deviation for Collective Anomaly threshold (default: 1.5).
            collective_window (int): Rolling window size for collective anomaly tracking (default: 12 steps = 60s for 5s sampling).

        Returns:
            dict: Threshold calibration parameters.
        """
        arr = np.asarray(errors, dtype=float)
        mean_err = float(np.mean(arr))
        std_err = float(np.std(arr))

        point_thresh = float(mean_err + point_sigma * std_err)
        collective_thresh = float(mean_err + collective_sigma * std_err)

        return {
            'point_threshold': point_thresh,
            'collective_threshold': collective_thresh,
            'point_sigma': point_sigma,
            'collective_sigma': collective_sigma,
            'mean_error': mean_err,
            'std_error': std_err,
            'collective_window': int(collective_window)
        }

    @staticmethod
    def classify_anomalies(errors, point_threshold, collective_threshold, collective_window=12):
        """
        Classifies sequence errors into Point Anomalies and Collective Anomalies.

        Args:
            errors (np.ndarray or list or pd.Series): Reconstruction error per sequence (evaluated at last timestep).
            point_threshold (float): Threshold for immediate point spikes.
            collective_threshold (float): Threshold for sustained collective deviations.
            collective_window (int): Number of consecutive steps to assess collective patterns.

        Returns:
            dict: Classification arrays and boolean flags.
        """
        err_series = pd.Series(np.asarray(errors, dtype=float))
        is_point = err_series > point_threshold
        rolling_err = err_series.rolling(window=collective_window, min_periods=1).mean()
        is_collective = (rolling_err > collective_threshold) & (~is_point)

        anomaly_type = []
        for p, c in zip(is_point, is_collective):
            if p:
                anomaly_type.append('point')
            elif c:
                anomaly_type.append('collective')
            else:
                anomaly_type.append('normal')

        return {
            'is_point': is_point.to_numpy(),
            'is_collective': is_collective.to_numpy(),
            'is_anomaly': (is_point | is_collective).to_numpy(),
            'rolling_error': rolling_err.to_numpy(),
            'anomaly_type': anomaly_type
        }

if __name__ == "__main__":
    # Quick test of the pipeline
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    values = np.random.rand(100, 1) * 100
    values[10:15] = np.nan # Simulate missing data

    df = pd.DataFrame(values, columns=['energy'], index=dates)

    prep = Preprocessor()
    df_clean = prep.clean_data(df)
    scaled = prep.fit_transform(df_clean)
    sequences = prep.create_sequences(scaled, time_steps=24)

    print(f"Original shape: {df.shape}")
    print(f"Cleaned NaNs: {df_clean.isna().sum().sum()}")
    print(f"Sequences shape: {sequences.shape}")
