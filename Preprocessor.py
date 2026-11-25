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

STATIONARY_TELEMETRY_COLUMNS = [
    'voltage',
    'current',
    'power',
    'power_factor'
]

CYCLICAL_FEATURE_COLUMNS = [
    'hour_sin',
    'hour_cos',
    'week_sin',
    'week_cos'
]

# Legacy 9-feature set (kept for backward compatibility with older saved default models)
MODEL_FEATURE_COLUMNS = TELEMETRY_FEATURE_COLUMNS + CYCLICAL_FEATURE_COLUMNS

# Stationary 8-feature set (excludes unbounded cumulative energy) for per-load models
STATIONARY_MODEL_FEATURE_COLUMNS = STATIONARY_TELEMETRY_COLUMNS + CYCLICAL_FEATURE_COLUMNS


def _resolve_timezone(tz):
    if tz is None:
        return DATABASE_TIMEZONE
    if isinstance(tz, str):
        return ZoneInfo(tz)
    return tz


class Preprocessor:
    """Preprocessor for Autoencoder Anomaly Detection."""
    def __init__(self, feature_range=(0, 1), clip=False):
        self.scaler = MinMaxScaler(feature_range=feature_range, clip=clip)
        self.is_fitted = False

    @staticmethod
    def compute_daily_energy(df):
        """
        Computes time-invariant daily accumulated energy (resets to 0 each midnight).
        Prevents cumulative drift in multi-week and long-term production deployments.
        """
        if not isinstance(df, pd.DataFrame):
            return df
        result = df.copy()
        if isinstance(result.index, pd.DatetimeIndex):
            dates = result.index.date
        elif 'datetime' in result.columns:
            dates = pd.to_datetime(result['datetime']).dt.date
        else:
            return result
        if 'energy' in result.columns:
            result['daily_energy'] = result.groupby(dates)['energy'].transform(lambda x: x - x.iloc[0])
        return result

    def load_data(self, filepath):
        """Loads data from a CSV file."""
        return pd.read_csv(filepath, parse_dates=True, index_col=0)

    def clean_data(self, df):
        """Handles missing values using linear interpolation and forward/backward filling."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not df.isnull().values.any():
            return df

        df_clean = df.copy()
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        df_clean = df_clean.ffill().bfill()
        return df_clean

    def add_cyclical_time_features(self, df, timezone=DATABASE_TIMEZONE):
        """Add time-of-day and day-of-week cyclical features in a consistent timezone."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        target_tz = _resolve_timezone(timezone)
        result = df.copy()
        if 'datetime' in result.columns:
            timestamps = pd.DatetimeIndex(pd.to_datetime(result.pop('datetime')))
        else:
            timestamps = pd.DatetimeIndex(pd.to_datetime(result.index))

        if timestamps.isna().any():
            raise ValueError("Telemetry contains invalid datetime values.")

        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(DATABASE_TIMEZONE)
        timestamps = timestamps.tz_convert(target_tz)

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
        """Fits the scaler to the data and returns normalized data."""
        transformed_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        return transformed_data

    def transform(self, data):
        """Normalizes data using the fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """Reverses normalization to get original values."""
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

        data = np.ascontiguousarray(data)
        n_samples, n_features = data.shape

        if n_samples < time_steps:
            return np.empty((0, time_steps, n_features))

        n_sequences = n_samples - time_steps + 1
        shape = (n_sequences, time_steps, n_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])

        return as_strided(data, shape=shape, strides=strides)

    def create_sequences_gap_aware(self, data, timestamps=None, time_steps=24, max_gap_seconds=15, energy_col_idx=None):
        """
        Generates sliding window sequences without bridging across time gaps or energy resets.

        Args:
            data (np.ndarray or pd.DataFrame): Normalized feature matrix or DataFrame.
            timestamps: Timestamps corresponding to rows. Auto-detected from DataFrame index if omitted.
            time_steps (int): Length of the input sequence (lookback).
            max_gap_seconds (float): Max allowed gap between consecutive samples.
            energy_col_idx (int or None): Index of energy column to prevent bridging across reset cliffs.
                Defaults to None unless an 'energy' column is present in a DataFrame.

        Returns:
            tuple: (sequences_array, target_timestamps_list)
        """
        detected_energy_idx = energy_col_idx
        if isinstance(data, pd.DataFrame):
            if detected_energy_idx is None and 'energy' in data.columns:
                detected_energy_idx = data.columns.get_loc('energy')
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

        # Break sequence if an energy reset cliff (drop in cumulative energy) occurs
        if detected_energy_idx is not None and matrix.shape[1] > detected_energy_idx:
            energy_col = matrix[:, detected_energy_idx]
            energy_diffs = np.diff(energy_col, prepend=energy_col[0])
            is_break = is_break | (energy_diffs < -1e-4)

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
    def fit_schedule_profile(df, slot_minutes=15, on_power_threshold=1.0, timezone=DATABASE_TIMEZONE, per_slot_power_cap=True):
        """
        Learns a data-driven 7-day x (1440/slot_minutes) temporal profile from normal training data.
        Captures expected ON probability and typical ON power ceiling per (weekday, time-slot),
        dilated by +-1 adjacent slot to tolerate normal schedule edge jitter.
        When per_slot_power_cap=False, all ON slots share the global ON 99th-percentile ceiling.
        """
        target_tz = _resolve_timezone(timezone)
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif 'datetime' in df.columns:
            timestamps = pd.DatetimeIndex(pd.to_datetime(df['datetime']))
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'datetime' column.")

        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(DATABASE_TIMEZONE)
        timestamps = timestamps.tz_convert(target_tz)

        slots_per_day = 1440 // slot_minutes
        total_slots = 7 * slots_per_day
        slot_ids = timestamps.dayofweek.to_numpy() * slots_per_day + (
            (timestamps.hour.to_numpy() * 60 + timestamps.minute.to_numpy()) // slot_minutes
        )
        powers = pd.to_numeric(df['power'], errors='coerce').fillna(0.0).to_numpy(dtype=float)

        raw_on_prob = np.zeros(total_slots, dtype=float)
        raw_power_p95 = np.zeros(total_slots, dtype=float)

        for sid in range(total_slots):
            mask = slot_ids == sid
            if not np.any(mask):
                continue
            slot_p = powers[mask]
            on_mask = slot_p >= on_power_threshold
            raw_on_prob[sid] = float(np.mean(on_mask))
            if np.any(on_mask):
                raw_power_p95[sid] = float(np.percentile(slot_p[on_mask], 95))

        # +-1 slot neighbor dilation for smooth transition tolerance
        allowed_on_prob = [
            float(max(raw_on_prob[(i - 1) % total_slots], raw_on_prob[i], raw_on_prob[(i + 1) % total_slots]))
            for i in range(total_slots)
        ]
        if per_slot_power_cap:
            allowed_power_p95 = [
                float(max(raw_power_p95[(i - 1) % total_slots], raw_power_p95[i], raw_power_p95[(i + 1) % total_slots]))
                for i in range(total_slots)
            ]
        else:
            all_on = powers[powers >= on_power_threshold]
            global_on_p99 = float(np.percentile(all_on, 99)) if len(all_on) > 0 else 0.0
            allowed_power_p95 = [
                global_on_p99 if p >= 0.05 else 0.0
                for p in allowed_on_prob
            ]

        return {
            'slot_minutes': int(slot_minutes),
            'slots_per_day': int(slots_per_day),
            'timezone': str(target_tz),
            'on_power_threshold': float(on_power_threshold),
            'off_prob_threshold': 0.05,
            'over_power_factor': 1.35,
            'over_power_margin_w': 2.0,
            'allowed_on_prob': allowed_on_prob,
            'allowed_power_p95': allowed_power_p95,
        }

    @staticmethod
    def compute_schedule_penalty(timestamps, power_values, profile, penalty_scale):
        """
        Scores per-step deviations from the learned weekly schedule profile:
          - ON during a slot that was OFF in training -> penalty_scale
          - Sustained over-lighting above the slot's learned ceiling -> penalty_scale
          - OFF during a scheduled ON slot -> 0.0 (turning lights off saves energy)
        """
        if profile is None or len(timestamps) == 0:
            return np.zeros(len(timestamps), dtype=float)

        target_tz = _resolve_timezone(profile.get('timezone', DATABASE_TIMEZONE))
        ts_idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if ts_idx.tz is None:
            ts_idx = ts_idx.tz_localize(DATABASE_TIMEZONE)
        ts_idx = ts_idx.tz_convert(target_tz)

        slot_minutes = int(profile.get('slot_minutes', 15))
        slots_per_day = int(profile.get('slots_per_day', 1440 // slot_minutes))
        total_slots = 7 * slots_per_day
        slot_ids = (
            ts_idx.dayofweek.to_numpy() * slots_per_day
            + ((ts_idx.hour.to_numpy() * 60 + ts_idx.minute.to_numpy()) // slot_minutes)
        ) % total_slots

        allowed_on_prob = np.asarray(profile['allowed_on_prob'], dtype=float)
        allowed_power_p95 = np.asarray(profile['allowed_power_p95'], dtype=float)
        on_thresh = float(profile.get('on_power_threshold', 1.0))
        off_prob_thresh = float(profile.get('off_prob_threshold', 0.05))
        over_factor = float(profile.get('over_power_factor', 1.35))
        over_margin = float(profile.get('over_power_margin_w', 2.0))

        powers = np.asarray(power_values, dtype=float)
        slot_on_prob = allowed_on_prob[slot_ids]
        slot_p95 = allowed_power_p95[slot_ids]

        off_schedule_on = (slot_on_prob < off_prob_thresh) & (powers >= on_thresh)
        over_lighting = (
            (slot_on_prob >= off_prob_thresh)
            & (slot_p95 > on_thresh)
            & (powers > (slot_p95 * over_factor + over_margin))
        )

        return np.where(off_schedule_on | over_lighting, float(penalty_scale), 0.0)

    @staticmethod
    def compute_interval_energy(energy_series, fill_value=0.0):
        """
        Converts cumulative energy to stationary interval energy (delta kWh per sample).
        Negative steps (from hardware resets) are clipped to fill_value.
        """
        s = pd.Series(energy_series)
        delta = s.diff().fillna(fill_value)
        delta[delta < 0] = fill_value
        return delta.to_numpy()

    @staticmethod
    def to_window_relative_energy(sequences, energy_col_idx=3):
        """
        Transforms cumulative energy within each sequence to window-relative energy (E_t - E_0).
        This guarantees bounded, stationary energy features across arbitrarily long monitoring timeframes.
        """
        seqs = sequences.copy()
        seqs[:, :, energy_col_idx] = seqs[:, :, energy_col_idx] - seqs[:, [0], energy_col_idx]
        return seqs

    @staticmethod
    def compute_anomaly_thresholds(errors, point_sigma=3.0, collective_sigma=1.5, collective_window=12):
        """
        Computes data-driven anomaly detection thresholds from baseline reconstruction errors.

        Args:
            errors (np.ndarray or list): 1D array of reconstruction errors on normal baseline data.
            point_sigma (float): Multiplier of standard deviation for Point Anomaly threshold.
            collective_sigma (float): Multiplier of standard deviation for Collective Anomaly threshold.
            collective_window (int): Rolling window size for collective anomaly tracking.

        Returns:
            dict: Threshold calibration parameters.
        """
        arr = np.asarray(errors, dtype=float)
        mean_err = float(np.mean(arr))
        std_err = float(np.std(arr))

        return {
            'point_threshold': float(mean_err + point_sigma * std_err),
            'collective_threshold': float(mean_err + collective_sigma * std_err),
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
            errors: Reconstruction error per sequence.
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
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    values = np.random.rand(100, 1) * 100
    values[10:15] = np.nan

    df = pd.DataFrame(values, columns=['energy'], index=dates)

    prep = Preprocessor()
    df_clean = prep.clean_data(df)
    scaled = prep.fit_transform(df_clean)
    sequences = prep.create_sequences(scaled, time_steps=24)

    print(f"Original shape: {df.shape}")
    print(f"Cleaned NaNs: {df_clean.isna().sum().sum()}")
    print(f"Sequences shape: {sequences.shape}")
