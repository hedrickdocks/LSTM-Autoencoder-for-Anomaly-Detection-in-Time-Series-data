from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import logging
from Preprocessor import MODEL_FEATURE_COLUMNS, Preprocessor

app = FastAPI(title="EDR AI Model Service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model, scaler, and thresholds
model = None
scaler = None
thresholds = None

# Paths (Assuming these will be placed in the same directory by the user)
MODEL_PATH = os.getenv("MODEL_PATH", "lstm_autoencoder.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "anomaly_thresholds.pkl")

@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler, thresholds
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
            logger.info(f"Loaded TensorFlow model from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Predictions will fail until provided.")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Loaded scikit-learn scaler from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}. Data normalization won't happen.")

        if os.path.exists(THRESHOLDS_PATH):
            thresholds = joblib.load(THRESHOLDS_PATH)
            logger.info(f"Loaded anomaly thresholds from {THRESHOLDS_PATH}: Point={thresholds.get('point_threshold')}, Collective={thresholds.get('collective_threshold')}")
        else:
            logger.warning(f"Thresholds file not found at {THRESHOLDS_PATH}. Using fallback defaults.")
            thresholds = {'point_threshold': 0.5, 'collective_threshold': 0.3, 'collective_window': 12}
            
    except Exception as e:
        logger.error(f"Error loading model/scaler/thresholds during startup: {e}")

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

@app.post("/predict")
async def predict_anomalies(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded. Check server logs.")
        
    try:
        # 1. Convert incoming JSON array to Pandas DataFrame
        records = [record.dict() for record in request.data]
        df = pd.DataFrame(records)
        
        # Keep track of timestamps/identifiers
        # Data from Node.js may be sorted backwards (newest first). Let's sort it chronologically.
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        
        identifiers = df[['datetime', 'agent']] if 'agent' in df.columns else df[['datetime']]
        
        # 2. Derive the same UTC cyclical features used during training.
        preprocessor = Preprocessor()
        features_df = preprocessor.add_cyclical_time_features(df)
        features_df = features_df[MODEL_FEATURE_COLUMNS].fillna(0.0)
        
        # 3. Preprocess: Normalize using the saved scaler
        normalized_data = scaler.transform(features_df)
        
        # SEQUENCE GENERATION for LSTM Autoencoder (Gap-Aware)
        TIME_STEPS = 24
        
        X_seq, target_timestamps = preprocessor.create_sequences_gap_aware(
            normalized_data,
            timestamps=df['datetime'],
            time_steps=TIME_STEPS,
            max_gap_seconds=15
        )
        
        if len(X_seq) == 0:
             # Not enough continuous data to form a sequence
             return {"predictions": [], "message": f"Need at least {TIME_STEPS} continuous points"}
        
        # 4. Inference: Run through TensorFlow Autoencoder
        reconstructions = model.predict(X_seq, verbose=0)
        
        # Calculate reconstruction error (Mean Squared Error per sequence)
        mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))
        
        # Calibrated threshold evaluation
        point_th = thresholds.get('point_threshold', 0.5) if thresholds else 0.5
        coll_th = thresholds.get('collective_threshold', 0.3) if thresholds else 0.3
        coll_win = thresholds.get('collective_window', 12) if thresholds else 12

        classification = Preprocessor.classify_anomalies(
            mse,
            point_threshold=point_th,
            collective_threshold=coll_th,
            collective_window=coll_win
        )
        
        # 5. Format results
        results = []
        default_agent = identifiers['agent'].iloc[0] if 'agent' in identifiers.columns else "unknown"
        
        for i in range(len(mse)):
            dt_val = target_timestamps[i]
            dt_str = dt_val.isoformat() if hasattr(dt_val, 'isoformat') else str(dt_val)
            
            results.append({
                "datetime": dt_str,
                "agent": default_agent,
                "reconstruction_error": float(mse[i]),
                "rolling_error": float(classification['rolling_error'][i]),
                "anomaly_type": classification['anomaly_type'][i],
                "is_point_anomaly": bool(classification['is_point'][i]),
                "is_collective_anomaly": bool(classification['is_collective'][i]),
                "is_anomaly": bool(classification['is_anomaly'][i])
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
        "scaler_loaded": scaler is not None
    }
