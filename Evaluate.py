import os
import sqlite3
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.losses import MeanSquaredError

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
        os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'lstm_autoencoder.h5')),
        os.getenv('SCALER_PATH', os.path.join(BASE_DIR, 'scaler.pkl')),
        os.getenv('THRESHOLDS_PATH', os.path.join(BASE_DIR, 'anomaly_thresholds.pkl')),
    )


def score_dataframe(df_raw, model, scaler, thresholds):
    """Run inference matching main.py logic (last-step point MSE + hybrid schedule profile)."""
    prep = Preprocessor()
    prep.scaler = scaler
    prep.is_fitted = True

    feature_columns = thresholds.get('feature_columns') if thresholds else None
    if not feature_columns:
        n_in = getattr(scaler, 'n_features_in_', 9)
        feature_columns = list(MODEL_FEATURE_COLUMNS if n_in == 9 else STATIONARY_MODEL_FEATURE_COLUMNS)
    tz_name = (thresholds.get('timezone') if thresholds else None) or (
        'UTC' if len(feature_columns) == 9 else str(DATABASE_TIMEZONE)
    )
    energy_col_idx = feature_columns.index('energy') if 'energy' in feature_columns else None

    df = df_raw.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
    else:
        df = df.sort_index()

    with_time = prep.add_cyclical_time_features(df, timezone=tz_name)
    raw_power_by_ts = (
        pd.Series(pd.to_numeric(with_time['power'], errors='coerce').fillna(0.0).to_numpy(dtype=float), index=with_time.index)
        .groupby(level=0)
        .last()
    )
    feat_df = with_time[feature_columns].fillna(0.0)
    scaled = prep.transform(feat_df)

    X_seq, target_timestamps = prep.create_sequences_gap_aware(
        scaled,
        timestamps=with_time.index,
        time_steps=TIME_STEPS,
        max_gap_seconds=15,
        energy_col_idx=energy_col_idx,
    )
    if len(X_seq) == 0:
        return None

    recon = model.predict(X_seq, batch_size=512, verbose=0)
    point_mse = np.mean(np.power(X_seq[:, -1, :] - recon[:, -1, :], 2), axis=1)

    point_th = float(thresholds.get('point_threshold', 0.5)) if thresholds else 0.5
    coll_th = float(thresholds.get('collective_threshold', 0.3)) if thresholds else 0.3
    coll_win = int(thresholds.get('collective_window', 12)) if thresholds else 12

    is_point = point_mse > point_th

    schedule_profile = thresholds.get('schedule_profile') if thresholds else None
    if schedule_profile:
        target_idx = pd.DatetimeIndex(target_timestamps)
        target_powers = raw_power_by_ts.reindex(target_idx).fillna(0.0).to_numpy()
        penalty_scale = float(thresholds.get('schedule_penalty_scale', coll_th * 2.0))
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
        .rolling(window=coll_win, min_periods=min(coll_win, 4))
        .mean()
        .fillna(0.0)
        .to_numpy()
    )
    is_collective = (rolling_err > coll_th) & (~is_point)
    is_anomaly = is_point | is_collective

    return {
        'target_timestamps': target_timestamps,
        'point_mse': point_mse,
        'rolling_err': rolling_err,
        'is_point': is_point,
        'is_collective': is_collective,
        'is_anomaly': is_anomaly,
    }


def _make_block(start_manila, n_steps, v, pf, bulbs, bulb_w=4.75, seed=1, standby_current=0.0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start_manila, periods=n_steps, freq='5s', tz=DATABASE_TIMEZONE)
    v_arr = v + rng.normal(0.0, 0.2, size=n_steps)
    if bulbs == 0:
        c_arr = np.clip(standby_current + rng.normal(0.0, 0.002, size=n_steps), 0.0, 0.04) if standby_current > 0 else np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        pf_arr = np.zeros(n_steps)
    else:
        pf_arr = np.clip(pf + rng.normal(0.0, 0.004, size=n_steps), 0.1, 1.0)
        p_arr = bulbs * bulb_w + rng.normal(0.0, max(0.12, bulbs * bulb_w * 0.005), size=n_steps)
        c_arr = p_arr / (v_arr * pf_arr)
    return pd.DataFrame({
        'datetime': ts,
        'voltage': np.round(v_arr, 3),
        'current': np.round(c_arr, 3),
        'power': np.round(p_arr, 3),
        'energy': np.full(n_steps, 0.050),
        'power_factor': np.round(pf_arr, 3),
    })


def run_lights_scenarios(model, scaler, thresholds):
    print("\n=== 7-Scenario Lights Evaluation Suite ===")
    results = []

    def check(name, df_case, expect_fn, detail_fn=None):
        out = score_dataframe(df_case, model, scaler, thresholds)
        assert out is not None, f"{name}: produced 0 sequences"
        ok, msg = expect_fn(out)
        status = "PASS" if ok else "FAIL"
        extra = detail_fn(out) if detail_fn else ""
        print(f"[{status}] {name:46s} | {msg}{(' | ' + extra) if extra else ''}")
        results.append(ok)

    # 1. Morning scheduled half-open ON (Mon 09:30 Manila, 2 bulbs ~9.5W)
    s1 = _make_block('2026-10-19 09:30:00', 80, v=227.5, pf=0.635, bulbs=2, seed=10)
    check(
        "1. Morning half-open ON (~9.5W, 227.5V)",
        s1,
        lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 2. Afternoon/evening scheduled ON (Mon 19:30 Manila: 3 bulbs ~14.2W and 4 bulbs ~19W at 229V)
    s2a = _make_block('2026-10-19 19:30:00', 80, v=229.0, pf=0.636, bulbs=3, seed=20)
    s2b = _make_block('2026-10-19 16:30:00', 80, v=222.5, pf=0.618, bulbs=4, seed=21)
    check(
        "2a. Evening 3-bulb ON (~14.2W, 229V)",
        s2a,
        lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )
    check(
        "2b. Afternoon 4-bulb ON (~19.0W, 222.5V)",
        s2b,
        lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 3. Scheduled OFF (Mon 22:30 Manila, I=0, P=0, PF=0.000)
    s3 = _make_block('2026-10-19 22:30:00', 80, v=227.0, pf=0.0, bulbs=0, seed=30)
    check(
        "3. Scheduled OFF (PF=0.000, 227V)",
        s3,
        lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 4a. Normal OFF -> 1-bulb -> 3-bulb turn-on transition (Mon 19:00 Manila, 24+ steps after turn-on)
    off_part = _make_block('2026-10-19 18:55:00', 40, v=228.0, pf=0.0, bulbs=0, seed=40)
    mid_part = _make_block('2026-10-19 18:58:20', 3, v=228.0, pf=0.635, bulbs=1, seed=41)
    on_part = _make_block('2026-10-19 18:58:35', 60, v=228.2, pf=0.636, bulbs=3, seed=42)
    s4 = pd.concat([off_part, mid_part, on_part], ignore_index=True)
    check(
        "4a. OFF -> 1-bulb -> 3-bulb turn-on transition",
        s4,
        lambda o: (np.sum(o['is_point']) == 0 and np.sum(o['is_anomaly']) == 0,
                   f"point={np.sum(o['is_point'])}, total_anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 4b. Normal ON -> OFF with mid-window PZEM packet (3-bulb -> 1.7W/0.31PF -> OFF)
    on_before = _make_block('2026-10-19 19:24:00', 35, v=227.3, pf=0.638, bulbs=3, seed=43)
    mid_drop = pd.DataFrame({
        'datetime': pd.date_range('2026-10-19 19:26:55', periods=1, freq='5s', tz=DATABASE_TIMEZONE),
        'voltage': [227.929], 'current': [0.010], 'power': [1.671], 'energy': [0.050], 'power_factor': [0.312],
    })
    off_after = _make_block('2026-10-19 19:27:00', 45, v=227.4, pf=0.0, bulbs=0, seed=44)
    s4b = pd.concat([on_before, mid_drop, off_after], ignore_index=True)
    check(
        "4b. ON -> OFF with mid-window PZEM packet",
        s4b,
        lambda o: (np.sum(o['is_point']) == 0 and np.sum(o['is_anomaly']) == 0,
                   f"point={np.sum(o['is_point'])}, total_anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 5. Electrical point anomalies: 50W surge and 185V sag
    s5_surge = _make_block('2026-10-19 19:30:00', 60, v=226.0, pf=0.630, bulbs=3, seed=50)
    s5_surge.loc[45, 'power'] = 50.0
    s5_surge.loc[45, 'current'] = 50.0 / (226.0 * 0.630)
    check(
        "5a. Electrical point spike (50W surge)",
        s5_surge,
        lambda o: (np.sum(o['is_point']) >= 1, f"point={np.sum(o['is_point'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )
    s5_sag = _make_block('2026-10-19 19:30:00', 60, v=226.0, pf=0.630, bulbs=3, seed=51)
    s5_sag.loc[45, 'voltage'] = 185.0
    check(
        "5b. Electrical point spike (185V voltage sag)",
        s5_sag,
        lambda o: (np.sum(o['is_point']) >= 1, f"point={np.sum(o['is_point'])}"),
        lambda o: f"max_mse={o['point_mse'].max():.5f}",
    )

    # 6. Off-schedule lights ON (Mon 23:00 Manila & Sun 14:00 Manila)
    s6a = _make_block('2026-10-19 23:00:00', 70, v=226.0, pf=0.635, bulbs=3, seed=60)
    s6b = _make_block('2026-10-25 14:00:00', 70, v=226.0, pf=0.635, bulbs=2, seed=61)
    check(
        "6a. Off-schedule ON (Mon 23:00 MNL, ~14W)",
        s6a,
        lambda o: (np.sum(o['is_collective']) >= 20, f"collective={np.sum(o['is_collective'])}/{len(o['is_collective'])}"),
        lambda o: f"max_roll={o['rolling_err'].max():.5f}",
    )
    check(
        "6b. Off-schedule ON (Sun 14:00 MNL, ~9.5W)",
        s6b,
        lambda o: (np.sum(o['is_collective']) >= 20, f"collective={np.sum(o['is_collective'])}/{len(o['is_collective'])}"),
        lambda o: f"max_roll={o['rolling_err'].max():.5f}",
    )

    # 7. Morning over-lighting (4 bulbs ~19W sustained during Mon 09:30 half-open block)
    s7 = _make_block('2026-10-19 09:30:00', 70, v=226.0, pf=0.630, bulbs=4, seed=70)
    check(
        "7. Morning over-lighting (4 bulbs ~19W @ 09:30)",
        s7,
        lambda o: (np.sum(o['is_collective']) >= 20, f"collective={np.sum(o['is_collective'])}/{len(o['is_collective'])}"),
        lambda o: f"max_roll={o['rolling_err'].max():.5f}",
    )

    passed = sum(results)
    print(f"\nScenario Summary: {passed}/{len(results)} passed.")
    return all(results)


def run_fan_scenarios(model, scaler, thresholds):
    print("\n=== Ceiling Fan (4-Fan) Evaluation Suite ===")
    results = []

    def check(name, df_case, expect_fn, detail_fn=None):
        out = score_dataframe(df_case, model, scaler, thresholds)
        assert out is not None, f"{name}: produced 0 sequences"
        ok, msg = expect_fn(out)
        status = "PASS" if ok else "FAIL"
        extra = detail_fn(out) if detail_fn else ""
        print(f"[{status}] {name:48s} | {msg}{(' | ' + extra) if extra else ''}")
        results.append(ok)

    # 1. Morning 1 fan Stage 1 (~40.3W)
    s1 = _make_block('2026-10-19 09:30:00', 80, v=226.0, pf=0.950, bulbs=1, bulb_w=40.3, seed=101)
    check("1. Morning 1 fan Stage 1 (~40W, PF=0.95)", s1,
          lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 2. Morning 2 fans Stage 1 (~80.6W)
    s2 = _make_block('2026-10-19 09:30:00', 80, v=226.0, pf=0.980, bulbs=2, bulb_w=40.3, seed=102)
    check("2. Morning 2 fans Stage 1 (~81W, PF=0.98)", s2,
          lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 3. Morning 4 fans Stage 3 (~192W) -> valid normal combination
    s3 = _make_block('2026-10-19 09:30:00', 80, v=226.0, pf=0.995, bulbs=4, bulb_w=48.0, seed=103)
    check("3. Morning 4 fans Stage 3 (~192W, PF=0.995)", s3,
          lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 4. Evening 3 fans Stage 3 (~142W, matching real local.db)
    s4 = _make_block('2026-10-19 19:30:00', 80, v=225.5, pf=1.000, bulbs=3, bulb_w=47.3, seed=104)
    check("4. Evening 3 fans Stage 3 (~142W, PF=1.00)", s4,
          lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 5. Scheduled OFF with standby CT noise (0W, PF=0.000, I~0.025A)
    s5 = _make_block('2026-10-19 22:30:00', 80, v=226.4, pf=0.0, bulbs=0, seed=105, standby_current=0.025)
    check("5. Scheduled OFF (0W, PF=0.000, I~0.025A)", s5,
          lambda o: (np.sum(o['is_anomaly']) == 0, f"anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 6a. Turn-on transition (OFF -> 3 fans Stage 3)
    off_p = _make_block('2026-10-19 18:55:00', 40, v=226.0, pf=0.0, bulbs=0, seed=106, standby_current=0.020)
    on_p = _make_block('2026-10-19 18:58:20', 60, v=225.8, pf=0.998, bulbs=3, bulb_w=47.5, seed=107)
    s6a = pd.concat([off_p, on_p], ignore_index=True)
    check("6a. Turn-on transition (OFF -> 3 fans Stg 3)", s6a,
          lambda o: (np.sum(o['is_point']) == 0 and np.sum(o['is_anomaly']) == 0,
                     f"point={np.sum(o['is_point'])}, total_anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 6b. Speed step & turn-off (3 fans Stg 3 -> Stg 1 -> mid-window -> OFF)
    st3 = _make_block('2026-10-19 19:22:00', 35, v=225.5, pf=0.998, bulbs=3, bulb_w=47.5, seed=108)
    st1 = _make_block('2026-10-19 19:24:55', 25, v=225.7, pf=0.980, bulbs=3, bulb_w=40.3, seed=109)
    mid = pd.DataFrame({
        'datetime': pd.date_range('2026-10-19 19:27:00', periods=1, freq='5s', tz=DATABASE_TIMEZONE),
        'voltage': [226.0], 'current': [0.220], 'power': [46.0], 'energy': [0.050], 'power_factor': [0.450],
    })
    off_after = _make_block('2026-10-19 19:27:05', 45, v=226.2, pf=0.0, bulbs=0, seed=110, standby_current=0.015)
    s6b = pd.concat([st3, st1, mid, off_after], ignore_index=True)
    check("6b. Speed step & turn-off transition", s6b,
          lambda o: (np.sum(o['is_point']) == 0 and np.sum(o['is_anomaly']) == 0,
                     f"point={np.sum(o['is_point'])}, total_anom={np.sum(o['is_anomaly'])}/{len(o['is_anomaly'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 7a. Electrical point spike: 350W locked-rotor surge
    s7a = _make_block('2026-10-19 19:30:00', 60, v=225.5, pf=0.990, bulbs=3, bulb_w=47.5, seed=111)
    s7a.loc[45, 'power'] = 350.0
    s7a.loc[45, 'current'] = 350.0 / (225.5 * 0.990)
    check("7a. Electrical point spike (350W surge)", s7a,
          lambda o: (np.sum(o['is_point']) >= 1, f"point={np.sum(o['is_point'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 7b. Electrical point spike: 185V voltage sag
    s7b = _make_block('2026-10-19 19:30:00', 60, v=225.5, pf=0.990, bulbs=3, bulb_w=47.5, seed=112)
    s7b.loc[45, 'voltage'] = 185.0
    check("7b. Electrical point spike (185V voltage sag)", s7b,
          lambda o: (np.sum(o['is_point']) >= 1, f"point={np.sum(o['is_point'])}"),
          lambda o: f"max_mse={o['point_mse'].max():.5f}")

    # 8a. Off-schedule ON (Mon 23:00 MNL, 2 fans ~81W)
    s8a = _make_block('2026-10-19 23:00:00', 70, v=225.5, pf=0.980, bulbs=2, bulb_w=40.5, seed=113)
    check("8a. Off-schedule ON (Mon 23:00 MNL, ~81W)", s8a,
          lambda o: (np.sum(o['is_collective']) >= 20, f"collective={np.sum(o['is_collective'])}/{len(o['is_collective'])}"),
          lambda o: f"max_roll={o['rolling_err'].max():.5f}")

    # 8b. Off-schedule ON (Sun 14:00 MNL, 4 fans ~190W)
    s8b = _make_block('2026-10-25 14:00:00', 70, v=225.5, pf=0.995, bulbs=4, bulb_w=47.5, seed=114)
    check("8b. Off-schedule ON (Sun 14:00 MNL, ~190W)", s8b,
          lambda o: (np.sum(o['is_collective']) >= 20, f"collective={np.sum(o['is_collective'])}/{len(o['is_collective'])}"),
          lambda o: f"max_roll={o['rolling_err'].max():.5f}")

    passed = sum(results)
    print(f"\nFan Scenario Summary: {passed}/{len(results)} passed.")
    return all(results)


def find_default_test_data(load=None):
    env_path = os.getenv('TEST_DATA_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        os.path.abspath(os.path.join(BASE_DIR, '..', 'backend', 'test_data.db')),
        os.path.abspath(os.path.join(BASE_DIR, 'datasets', 'Saturday_dataset.csv')),
        os.path.abspath(os.path.join(BASE_DIR, 'datasets', 'Friday_dataset.csv')),
        os.path.abspath(os.path.join(BASE_DIR, 'generated_energy_data.csv')),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_labeled_test_data(data_path=None, load=None, inject_anomalies=True):
    resolved_path = data_path or find_default_test_data(load=load)
    if resolved_path and not os.path.exists(resolved_path):
        alt_path = os.path.abspath(os.path.join(BASE_DIR, '..', resolved_path))
        if os.path.exists(alt_path):
            resolved_path = alt_path
    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Test data file not found at '{resolved_path}'.")

    print(f"Loading evaluation dataset from: {resolved_path}")
    if resolved_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
        conn = sqlite3.connect(resolved_path)
        if load:
            df = pd.read_sql_query(
                "SELECT datetime, voltage, current, power, energy, power_factor FROM load_readings WHERE load = ? ORDER BY datetime DESC LIMIT 1000",
                conn,
                params=(load,),
            )
            df = df.iloc[::-1].reset_index(drop=True)
        else:
            df = pd.read_sql_query(
                "SELECT datetime, voltage, current, power, energy, power_factor FROM energy_data ORDER BY datetime ASC LIMIT 1000",
                conn,
            )
        conn.close()
    else:
        df = pd.read_csv(resolved_path)
        if len(df) > 1000:
            df = df.iloc[:1000].copy()

    n_samples = len(df)
    if n_samples < 150:
        raise ValueError(f"Test data is too short ({n_samples} rows). Need at least 150 rows.")

    labels = np.zeros(n_samples, dtype=int)
    if inject_anomalies:
        print("Injecting synthetic point spikes at indices [40, 120, 240]...")
        for idx in [40, 120, 240]:
            if idx < n_samples:
                df.iloc[idx, df.columns.get_loc('power')] = 15000.0
                labels[idx] = 1

    return df, labels


def evaluate_model(data_path=None, load=None, inject_anomalies=True, run_scenarios=False, save_output=None):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()

    class _Tee(io.TextIOBase):
        def __init__(self, orig, mirror):
            self.orig = orig
            self.mirror = mirror
        def write(self, s):
            self.orig.write(s)
            self.mirror.write(s)
            return len(s)
        def flush(self):
            self.orig.flush()
            self.mirror.flush()

    import sys
    tee = _Tee(sys.stdout, buf)
    with redirect_stdout(tee):
        print(f"--- Starting Evaluation (load={load or 'default'}) ---")
        model_path, scaler_path, thresholds_path = resolve_artifact_paths(load)
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})
        scaler = joblib.load(scaler_path)
        thresholds = joblib.load(thresholds_path) if os.path.exists(thresholds_path) else {}
        print(f"Loaded thresholds: Point={thresholds.get('point_threshold')}, Collective={thresholds.get('collective_threshold')}")

        scenarios_ok = True
        if run_scenarios or load == 'lights':
            if load == 'fan':
                scenarios_ok = run_fan_scenarios(model, scaler, thresholds)
            else:
                scenarios_ok = run_lights_scenarios(model, scaler, thresholds)
        elif load == 'fan':
            scenarios_ok = run_fan_scenarios(model, scaler, thresholds)

        df, raw_labels = get_labeled_test_data(data_path, load=load, inject_anomalies=inject_anomalies)
        out = score_dataframe(df, model, scaler, thresholds)
        if out is None:
            print("No valid continuous sequences found.")
        else:
            ts_to_label = dict(zip(pd.to_datetime(df['datetime'], utc=True), raw_labels))
            y_true_point = np.array([ts_to_label.get(pd.Timestamp(ts).tz_convert('UTC'), 0) for ts in out['target_timestamps']])
            y_pred_point = out['is_point'].astype(int)

            print("\n--- Point Anomaly Metrics (Last-Step MSE > Point Threshold) ---")
            print(classification_report(y_true_point, y_pred_point, labels=[0, 1], target_names=['Normal', 'Anomaly'], zero_division=0))
            print("Confusion Matrix:\n", confusion_matrix(y_true_point, y_pred_point, labels=[0, 1]))
            print(f"Identified Point: {np.sum(out['is_point'])}, Collective: {np.sum(out['is_collective'])}")

    if save_output:
        os.makedirs(os.path.dirname(os.path.abspath(save_output)), exist_ok=True)
        with open(save_output, 'w', encoding='utf-8') as f:
            f.write(buf.getvalue())
        print(f"Saved evaluation output -> {save_output}")
    return scenarios_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM Autoencoder.")
    parser.add_argument("--data", default=None, help="Path to test CSV or SQLite database file")
    parser.add_argument("--load", choices=["lights", "fan", "outlet", "aircon"], default=None)
    parser.add_argument("--no-inject", action="store_true", help="Evaluate raw data without injecting synthetic anomalies")
    parser.add_argument("--scenarios", action="store_true", help="Run the verification scenario suite")
    parser.add_argument("--save-output", default=None, help="Optional file path to save the evaluation report")
    args = parser.parse_args()
    ok = evaluate_model(
        data_path=args.data,
        load=args.load,
        inject_anomalies=not args.no_inject,
        run_scenarios=args.scenarios,
        save_output=args.save_output,
    )
    if not ok:
        raise SystemExit(1)
