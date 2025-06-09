import joblib
import numpy as np
import pandas as pd
import logging
from typing import Union, Dict

class PredictionService:
    model = None

    @staticmethod
    def load_model(model_path: str):
        """
        Loads a model from disk (e.g., a joblib .pkl file).
        """
        try:
            PredictionService.model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise

    @staticmethod
    def predict_ecg_classification(summary: dict, age: int, gender: str):
        """
        Predict ECG class from extracted features.
        `features` can be a dict (with named features) or numpy array.
        """
        if PredictionService.model is None:
            raise RuntimeError("Model is not loaded. Call load_model first.")

        try:
            expected_features = list(PredictionService.model.feature_names_in_)

            gender = gender.lower() if gender else "o"
            gender_map = {"m": 0, "f": 1, "o": 2}
            gender_encoded = gender_map.get(gender, 2)

            feature_dict = {
                "heart_rate": summary.get("heart_rate"),
                "p_duration": summary.get("intervals", {}).get("P_wave_duration_ms"),
                "pq_interval": summary.get("intervals", {}).get("PQ_interval_ms"),
                "qrs_duration": summary.get("intervals", {}).get("QRS_duration_ms"),
                "qt_interval": summary.get("intervals", {}).get("QT_interval_ms"),
                "age": age,
                "gender_encoded": gender_encoded,
            }

            physionet_features = summary.get("physionet_features", {})
            # Add all physionet features expected by the model, if they exist
            for feat in ["rr_interval", "p_axis", "qrs_axis", "t_axis"]:
                feature_dict[feat] = physionet_features.get(feat, None)

            df = pd.DataFrame([feature_dict])

            missing = set(expected_features) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

            df = df[expected_features]

            y_pred = PredictionService.model.predict(df)[0]
            y_prob = float(PredictionService.model.predict_proba(df)[0][1])

            summary["classification"] = "Abnormal" if y_pred == 1 else "Normal"
            summary["confidence"] = round(y_prob, 4)

            return summary

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
