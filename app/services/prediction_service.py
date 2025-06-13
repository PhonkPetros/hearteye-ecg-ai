import joblib
import numpy as np
import pandas as pd
import logging
from typing import Union, Dict
from ..utils.exceptions import APIError, ErrorCodes
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)

class PredictionService:
    model = None 

    @staticmethod
    def load_model(model_path: str):
        """
        Loads a machine learning model from disk (e.g., a joblib .pkl file).
        This method should typically be called once at application startup.
        """
        logger.info(f"Attempting to load ML model from: {model_path}")
        try:
            PredictionService.model = joblib.load(model_path)
            logger.info(f"ML model loaded successfully from: {model_path}")
        except FileNotFoundError:
            # If the model file itself isn't found
            logger.error(f"Model file not found at: {model_path}")
            raise APIError(
                ErrorCodes.RECORD_NOT_FOUND,
                f"Prediction model file not found at {model_path}.",
                status_code=500, # Treat as internal server error since model is essential
                details={"model_path": model_path}
            )
        except Exception as e:
            # Catch any other general exceptions during model loading (e.g., corrupted file)
            logger.exception(f"Failed to load prediction model from {model_path}: {e}")
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "Failed to load prediction model due to an unexpected error.",
                status_code=500,
                details={"model_path": model_path, "error_type": type(e).__name__, "message": str(e)}
            )

    @staticmethod
    def predict_ecg_classification(summary: dict, age: int, gender: str) -> Dict[str, Union[str, float]]:
        """
        Predict ECG class (Normal/Abnormal) from extracted features using the loaded ML model.
        Args:
            summary (dict): Dictionary containing ECG features, including "heart_rate", "intervals", and "physionet_features".
            age (int): Patient's age.
            gender (str): Patient's gender ('M', 'F', 'O').
        Returns:
            Dict[str, Union[str, float]]: The original summary dict with 'classification' and 'confidence' added.
        Raises:
            APIError: If the model is not loaded, required features are missing, or prediction fails.
        """
        if PredictionService.model is None:
            logger.error("Prediction model is not loaded when predict_ecg_classification was called.")
            raise APIError(
                ErrorCodes.SERVICE_UNAVAILABLE,
                "Prediction model is not loaded. Please ensure the model is loaded at application startup.",
                status_code=500
            )

        logger.info(f"Attempting ECG classification prediction for age={age}, gender={gender}",
                    extra={"summary_keys": list(summary.keys()), "age": age, "gender": gender})
        try:
            expected_features = list(PredictionService.model.feature_names_in_)

            # Standardize gender input and encode
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
            
            # Convert dictionary to DataFrame for prediction
            df = pd.DataFrame([feature_dict])
            df = df.apply(pd.to_numeric, errors='coerce')

            # Check for missing features required by the model
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                logger.error(f"Missing required features for prediction: {missing_features}",
                             extra={"missing_features": list(missing_features), "provided_keys": list(feature_dict.keys())})
                raise APIError(
                    ErrorCodes.INVALID_FORMAT,
                    f"Missing features required for prediction: {', '.join(missing_features)}",
                    status_code=400,
                    details={"missing_features": list(missing_features)}
                )
            
            # Ensure features are in the correct order as expected by the model
            df = df[expected_features]

            # Perform prediction
            y_pred = PredictionService.model.predict(df)[0]
            y_prob = float(PredictionService.model.predict_proba(df)[0][y_pred])  # Confidence of predicted class

            # Update the summary with classification results
            summary["classification"] = "Abnormal" if y_pred == 1 else "Normal"
            summary["confidence"] = round(y_prob, 4)

            logger.info(f"ECG classification completed: {summary['classification']} with confidence {summary['confidence']}",
                        extra={"classification": summary["classification"], "confidence": summary["confidence"], "patient_age": age})
            return summary

        except APIError:
            raise
        except ValueError as e:
            # Catch ValueErrors that might indicate malformed data or unexpected numerical issues
            logger.warning(f"Value error during prediction processing: {e}", exc_info=e)
            raise APIError(
                ErrorCodes.INVALID_FORMAT,
                f"Invalid data format or value for prediction: {str(e)}",
                status_code=400,
                details={"error_detail": str(e)}
            )
        except Exception as e:
            # Catch any other unexpected errors during the prediction process
            logger.exception(f"An unexpected error occurred during ECG classification prediction: {e}")
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "An unexpected error occurred during ECG classification.",
                status_code=500,
                details={"error_type": type(e).__name__, "message": str(e)}
            )