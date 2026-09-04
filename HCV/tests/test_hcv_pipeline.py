"""
Automated Test Suite for UCI HCV Pipeline & Artifact Reproducibility
---------------------------------------------------------------------
"""

import os
import sys
import unittest
import joblib
import pandas as pd
import numpy as np

# Ensure HCV directory is in sys.path
HCV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HCV_DIR)

from src.inference.predict_hcv import predict_patient_hcv, find_artifact_file


class TestHCVPipeline(unittest.TestCase):

    def test_artifact_existence(self):
        """Verify that all required model and preprocessing artifacts exist."""
        required_artifacts = [
            ("best_hcv_ml_model.joblib", "models"),
            ("hcv_qml_model_weights.pt", "models"),
            ("hcv_scaler.joblib", "preprocessing"),
            ("hcv_pca_4.joblib", "preprocessing"),
            ("hcv_imputer.joblib", "preprocessing"),
            ("hcv_sex_encoder.joblib", "preprocessing"),
            ("hcv_metrics.json", "metrics"),
        ]
        for fname, sub in required_artifacts:
            path = find_artifact_file(fname, sub)
            self.assertTrue(os.path.exists(path), f"Artifact missing: {fname}")

    def test_ml_inference_healthy(self):
        """Test classical ML inference on a healthy patient profile."""
        sample_healthy = {
            "Age": 35.0, "Sex": "m", "ALB": 42.0, "ALP": 55.0,
            "ALT": 20.0, "AST": 22.0, "BIL": 9.0, "CHE": 9.5,
            "CHOL": 5.0, "CREA": 75.0, "GGT": 18.0, "PROT": 74.0
        }
        res = predict_patient_hcv(sample_healthy, model_type="ml")
        self.assertEqual(res["prediction_class"], 0)
        self.assertEqual(res["prediction_label"], "Healthy/Control")
        self.assertGreater(res["confidence"], 0.70)

    def test_ml_inference_pathology(self):
        """Test classical ML inference on an HCV pathology patient profile."""
        sample_pathology = {
            "Age": 55.0, "Sex": "m", "ALB": 30.0, "ALP": 120.0,
            "ALT": 150.0, "AST": 210.0, "BIL": 55.0, "CHE": 4.5,
            "CHOL": 3.8, "CREA": 110.0, "GGT": 250.0, "PROT": 62.0
        }
        res = predict_patient_hcv(sample_pathology, model_type="ml")
        self.assertEqual(res["prediction_class"], 1)
        self.assertEqual(res["prediction_label"], "HCV-related pathology")
        self.assertGreater(res["confidence"], 0.70)

    def test_qml_inference(self):
        """Test PennyLane QML inference."""
        sample_patient = {
            "Age": 45.0, "Sex": "f", "ALB": 38.0, "ALP": 60.0,
            "ALT": 30.0, "AST": 35.0, "BIL": 12.0, "CHE": 8.0,
            "CHOL": 4.8, "CREA": 70.0, "GGT": 22.0, "PROT": 70.0
        }
        res = predict_patient_hcv(sample_patient, model_type="qml")
        self.assertIn("prediction_class", res)
        self.assertIn(res["prediction_class"], [0, 1])
        self.assertIn("confidence", res)


if __name__ == "__main__":
    unittest.main()
