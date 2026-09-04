"""
Automated Test Suite for Heart Disease (CAD) Pipeline & Artifact Reproducibility
--------------------------------------------------------------------------------
"""

import os
import sys
import unittest
import joblib
import pandas as pd
import numpy as np

HEART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HEART_DIR)

from src.inference.predict_heart import predict_patient_heart, find_artifact_file


class TestHeartPipeline(unittest.TestCase):

    def test_artifact_existence(self):
        """Verify that all required model and preprocessing artifacts exist."""
        required_artifacts = [
            ("best_heart_ml_model.joblib", "models"),
            ("heart_qml_model_weights.pt", "models"),
            ("heart_scaler.joblib", "preprocessing"),
            ("heart_pca_4.joblib", "preprocessing"),
            ("heart_imputer.joblib", "preprocessing"),
            ("heart_metrics.json", "metrics"),
        ]
        for fname, sub in required_artifacts:
            path = find_artifact_file(fname, sub)
            self.assertTrue(os.path.exists(path), f"Artifact missing: {fname}")

    def test_ml_inference_low_risk(self):
        """Test classical ML inference on a low-risk/absence profile."""
        sample_normal = {
            "age": 42.0, "sex": 0.0, "cp": 2.0, "trestbps": 120.0,
            "chol": 180.0, "fbs": 0.0, "restecg": 0.0, "thalach": 172.0,
            "exang": 0.0, "oldpeak": 0.0, "slope": 1.0, "ca": 0.0,
            "thal": 3.0
        }
        res = predict_patient_heart(sample_normal, model_type="ml")
        self.assertEqual(res["prediction_class"], 0)
        self.assertIn("Absence", res["prediction_label"])
        self.assertGreater(res["confidence"], 0.70)

    def test_ml_inference_high_risk(self):
        """Test classical ML inference on a high-risk/presence profile."""
        sample_pathology = {
            "age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0,
            "chol": 286.0, "fbs": 0.0, "restecg": 2.0, "thalach": 108.0,
            "exang": 1.0, "oldpeak": 2.6, "slope": 2.0, "ca": 3.0,
            "thal": 7.0
        }
        res = predict_patient_heart(sample_pathology, model_type="ml")
        self.assertEqual(res["prediction_class"], 1)
        self.assertIn("Presence", res["prediction_label"])
        self.assertGreater(res["confidence"], 0.70)

    def test_qml_inference(self):
        """Test PennyLane QML inference execution."""
        sample_patient = {
            "age": 55.0, "sex": 1.0, "cp": 3.0, "trestbps": 130.0,
            "chol": 240.0, "fbs": 0.0, "restecg": 1.0, "thalach": 150.0,
            "exang": 0.0, "oldpeak": 1.0, "slope": 2.0, "ca": 0.0,
            "thal": 3.0
        }
        res = predict_patient_heart(sample_patient, model_type="qml")
        self.assertIn("prediction_class", res)
        self.assertIn(res["prediction_class"], [0, 1])
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 0.50)

    def test_missing_feature_handling(self):
        """Ensure missing mandatory features trigger KeyError."""
        incomplete_sample = {
            "age": 50.0, "sex": 1.0, "cp": 2.0
        }
        with self.assertRaises(KeyError):
            predict_patient_heart(incomplete_sample, model_type="ml")

    def test_out_of_bounds_validation(self):
        """Ensure physiological out-of-bounds values trigger ValueError."""
        invalid_sample = {
            "age": 250.0, "sex": 1.0, "cp": 2.0, "trestbps": 120.0,  # age 250 is impossible
            "chol": 180.0, "fbs": 0.0, "restecg": 0.0, "thalach": 172.0,
            "exang": 0.0, "oldpeak": 0.0, "slope": 1.0, "ca": 0.0,
            "thal": 3.0
        }
        with self.assertRaises(ValueError):
            predict_patient_heart(invalid_sample, model_type="ml")

    def test_reproducibility_on_saved_test_set(self):
        """Verify that evaluation on untouched test set precisely reproduces logged metrics."""
        from sklearn.metrics import accuracy_score, f1_score
        import json

        metrics_file = find_artifact_file("heart_metrics.json", "metrics")
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        best_ml_name = metrics_data["best_classical_ml"]
        expected_acc = metrics_data["models_performance"][best_ml_name]["Accuracy"]
        expected_mac_f1 = metrics_data["models_performance"][best_ml_name]["Macro F1"]

        ml_model = joblib.load(find_artifact_file("best_heart_ml_model.joblib", "models"))
        X_test_path = os.path.join(HEART_DIR, "data", "processed", "cleaned_data", "X_test.csv")
        y_test_path = os.path.join(HEART_DIR, "data", "processed", "cleaned_data", "y_test.csv")

        X_te = pd.read_csv(X_test_path).values
        y_te = pd.read_csv(y_test_path).values.ravel()

        y_pred = ml_model.predict(X_te)
        reprod_acc = accuracy_score(y_te, y_pred)
        reprod_mac_f1 = f1_score(y_te, y_pred, average="macro")

        self.assertAlmostEqual(reprod_acc, expected_acc, places=5)
        self.assertAlmostEqual(reprod_mac_f1, expected_mac_f1, places=5)


if __name__ == "__main__":
    unittest.main()
