from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import logging
from Preprocessor import (
    DATABASE_TIMEZONE,
    MODEL_FEATURE_COLUMNS,
    Preprocessor,
    STATIONARY_MODEL_FEATURE_COLUMNS,
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EnerTrack AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model, scaler and thresholds. "default" is the original model (trained on room/lights data);
# a load can have its own in models/<load>/ (train with: python train_from_db.py --load aircon).
# Loads differ a lot (an aircon draws ~1 kW, lights ~0.3 kW), so one model can't judge them all.
ARTIFACTS = {}
model = None
scaler = None
thresholds = None
FALLBACK_THRESHOLDS = {'point_threshold': 0.5, 'collective_threshold': 0.3, 'collective_window': 12}

# Absolute paths based on this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "lstm_autoencoder.h5"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(BASE_DIR, "scaler.pkl"))
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", os.path.join(BASE_DIR, "anomaly_thresholds.pkl"))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))


def load_artifacts(model_path, scaler_path, thresholds_path, name):
    """Loads one model + scaler + thresholds set; returns None if the model or scaler is missing."""
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    loaded_scaler = joblib.load(scaler_path)
    loaded_thresholds = joblib.load(thresholds_path) if os.path.exists(thresholds_path) else dict(FALLBACK_THRESHOLDS)
    logger.info(
        f"Loaded model '{name}': Point={loaded_thresholds.get('point_threshold')}, "
        f"Collective={loaded_thresholds.get('collective_threshold')}, "
        f"Features={len(loaded_thresholds.get('feature_columns') or []) or getattr(loaded_scaler, 'n_features_in_', 'unknown')}"
    )
    return {"model": loaded_model, "scaler": loaded_scaler, "thresholds": loaded_thresholds}


@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler, thresholds
    try:
        default = load_artifacts(MODEL_PATH, SCALER_PATH, THRESHOLDS_PATH, "default")
        if default:
            ARTIFACTS["default"] = default
            model, scaler, thresholds = default["model"], default["scaler"], default["thresholds"]
        else:
            logger.warning(f"Default model or scaler not found ({MODEL_PATH}, {SCALER_PATH}). Predictions will fail until provided.")
            thresholds = dict(FALLBACK_THRESHOLDS)

        if os.path.isdir(MODELS_DIR):
            for load in sorted(os.listdir(MODELS_DIR)):
                folder = os.path.join(MODELS_DIR, load)
                if not os.path.isdir(folder):
                    continue
                artifacts = load_artifacts(
                    os.path.join(folder, "lstm_autoencoder.h5"),
                    os.path.join(folder, "scaler.pkl"),
                    os.path.join(folder, "anomaly_thresholds.pkl"),
                    load,
                )
                if artifacts:
                    ARTIFACTS[load] = artifacts
    except Exception as e:
        logger.error(f"Error loading model/scaler/thresholds during startup: {e}")


def artifacts_for(load):
    """Model set for a load. No load (room totals) and lights fall back to the default model,
    which was trained on lights; other loads need their own trained model."""
    if load and load in ARTIFACTS:
        return ARTIFACTS[load]
    if not load or load in ("room", "lights"):
        return ARTIFACTS.get("default")
    return None


def resolve_artifact_config(chosen):
    """Resolve feature columns, timezone, and energy column index for legacy or per-load models."""
    sc = chosen["scaler"]
    th = chosen["thresholds"] or {}
    feature_columns = th.get("feature_columns")
    if not feature_columns:
        n_in = getattr(sc, "n_features_in_", 9)
        feature_columns = list(MODEL_FEATURE_COLUMNS if n_in == 9 else STATIONARY_MODEL_FEATURE_COLUMNS)
    tz_name = th.get("timezone") or ('UTC' if len(feature_columns) == 9 else str(DATABASE_TIMEZONE))
    energy_col_idx = feature_columns.index('energy') if 'energy' in feature_columns else None
    return feature_columns, tz_name, energy_col_idx


# Data structure expected from Node.js backend
class SensorPayload(BaseModel):
    datetime: str
    agent: str | None = None
    voltage: float
    current: float
    power: float
    energy: float
    power_factor: float


class PredictionRequest(BaseModel):
    data: list[SensorPayload]
    load: str | None = None  # lights, fan, outlet, aircon; omitted for room totals


@app.post("/predict")
async def predict_anomalies(request: PredictionRequest):
    chosen = artifacts_for(request.load)
    if chosen is None:
        if request.load and request.load not in ("room", "lights"):
            # Not an error: this load just has no model trained yet.
            return {"predictions": [], "skipped": f"No model trained for load '{request.load}'"}
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded. Check server logs.")
    active_model, active_scaler, active_thresholds = chosen["model"], chosen["scaler"], chosen["thresholds"]

    try:
        feature_columns, tz_name, energy_col_idx = resolve_artifact_config(chosen)

        # 1. Convert incoming JSON array to Pandas DataFrame sorted chronologically
        records = [record.model_dump() for record in request.data]
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

        identifiers = df[['datetime', 'agent']] if 'agent' in df.columns else df[['datetime']]

        # 2. Derive cyclical features in the artifact's training timezone
        preprocessor = Preprocessor()
        features_with_time = preprocessor.add_cyclical_time_features(df, timezone=tz_name)
        raw_power_by_ts = (
            pd.Series(pd.to_numeric(df['power'], errors='coerce').fillna(0.0).to_numpy(dtype=float), index=features_with_time.index)
            .groupby(level=0)
            .last()
        )
        features_df = features_with_time[feature_columns].fillna(0.0)

        # 3. Preprocess: Normalize using the saved scaler
        normalized_data = active_scaler.transform(features_df)

        # SEQUENCE GENERATION for LSTM Autoencoder (Gap-Aware)
        TIME_STEPS = 24
        X_seq, target_timestamps = preprocessor.create_sequences_gap_aware(
            normalized_data,
            timestamps=features_with_time.index,
            time_steps=TIME_STEPS,
            max_gap_seconds=15,
            energy_col_idx=energy_col_idx,
        )

        if len(X_seq) == 0:
            return {"predictions": [], "message": f"Need at least {TIME_STEPS} continuous points"}

        # 4. Inference: Run through TensorFlow Autoencoder
        reconstructions = active_model.predict(X_seq, verbose=0)

        # Calculate reconstruction errors:
        # A) Target-step point error: reconstruction MSE of the current measurement at index -1
        point_mse = np.mean(np.power(X_seq[:, -1, :] - reconstructions[:, -1, :], 2), axis=1)

        # B) Whole-window sequence error: for diagnostic visibility
        window_mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))

        # Calibrated threshold evaluation
        point_th = active_thresholds.get('point_threshold', 0.5) if active_thresholds else 0.5
        coll_th = active_thresholds.get('collective_threshold', 0.3) if active_thresholds else 0.3
        coll_win = active_thresholds.get('collective_window', 12) if active_thresholds else 12

        # Point Anomaly is strictly triggered when the current electrical/waveform reading spikes
        is_point = point_mse > point_th

        # Hybrid collective scoring: combine autoencoder error with learned temporal schedule profile (if present)
        schedule_profile = active_thresholds.get('schedule_profile') if active_thresholds else None
        if schedule_profile:
            target_idx = pd.DatetimeIndex(target_timestamps)
            target_powers = raw_power_by_ts.reindex(target_idx).fillna(0.0).to_numpy()
            penalty_scale = float(active_thresholds.get('schedule_penalty_scale', coll_th * 2.0))
            schedule_penalty = Preprocessor.compute_schedule_penalty(
                target_idx,
                target_powers,
                schedule_profile,
                penalty_scale,
            )
            collective_signal = np.maximum(point_mse, schedule_penalty)
        else:
            collective_signal = point_mse

        # Ignore isolated point spikes so a single surge doesn't create ghost collective anomalies
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

        anomaly_types = []
        for p, c in zip(is_point, is_collective):
            if p:
                anomaly_types.append('point')
            elif c:
                anomaly_types.append('collective')
            else:
                anomaly_types.append('normal')

        # 5. Format results
        results = []
        default_agent = identifiers['agent'].iloc[0] if 'agent' in identifiers.columns else "unknown"

        for i in range(len(point_mse)):
            dt_val = target_timestamps[i]
            dt_str = dt_val.isoformat() if hasattr(dt_val, 'isoformat') else str(dt_val)

            results.append({
                "datetime": dt_str,
                "agent": default_agent,
                "reconstruction_error": float(point_mse[i]),
                "window_error": float(window_mse[i]),
                "rolling_error": float(rolling_err[i]),
                "anomaly_type": anomaly_types[i],
                "is_point_anomaly": bool(is_point[i]),
                "is_collective_anomaly": bool(is_collective[i]),
                "is_anomaly": bool(is_anomaly[i]),
            })

        return {"predictions": results}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "up",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.get("/models")
def list_models():
    """Which loads have their own trained model ("default" covers room totals and lights)."""
    return {"models": sorted(ARTIFACTS.keys())}


@app.get("/thresholds")
def get_thresholds(load: str | None = None):
    """Anomaly thresholds in use, so clients can show scores relative to them."""
    chosen = artifacts_for(load)
    th = chosen["thresholds"] if chosen else (thresholds if not load else None)
    if not th:
        raise HTTPException(status_code=503, detail="Thresholds not loaded")
    return {
        "point_threshold": float(th.get("point_threshold")),
        "collective_threshold": float(th.get("collective_threshold")),
        "collective_window": int(th.get("collective_window", 12)),
    }
