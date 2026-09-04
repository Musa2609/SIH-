"""
Audited HCV Inference Engine (Module: HCV/src/inference/predict_hcv.py)
-------------------------------------------------------------------------
Accepts patient clinical biomarkers and predicts whether the patient is
Healthy/Control or exhibits HCV-related liver pathology, using saved pipeline transformers
and the top-performing classical ML (XGBoost/Random Forest) or QML model.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

CLASS_MAPPING = {
    0: "Healthy/Control",
    1: "HCV-related pathology"
}

FEATURE_COLS = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]


def find_artifact_file(filename, subfolder=None):
    """
    Search for artifact in subfolder first, then root artifacts folder.
    """
    if subfolder:
        sub_path = os.path.join(ARTIFACTS_DIR, subfolder, filename)
        if os.path.exists(sub_path):
            return sub_path
    root_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    raise FileNotFoundError(f"Artifact '{filename}' not found in {ARTIFACTS_DIR} (subfolder: {subfolder})")


def predict_patient_hcv(patient_data: dict, model_type: str = "ml") -> dict:
    """
    Predict HCV status for a single patient input dictionary.
    
    Parameters:
        patient_data (dict): Dictionary with keys for all 12 biomarkers:
                             ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        model_type (str): 'ml' for classical ML model (XGBoost), 'qml' for 4-qubit Quantum Classifier.
    """
    imputer_path = find_artifact_file("hcv_imputer.joblib", "preprocessing")
    sex_enc_path = find_artifact_file("hcv_sex_encoder.joblib", "preprocessing")
    scaler_path = find_artifact_file("hcv_scaler.joblib", "preprocessing")
    ml_model_path = find_artifact_file("best_hcv_ml_model.joblib", "models")

    imputer = joblib.load(imputer_path)
    sex_encoder = joblib.load(sex_enc_path)
    scaler = joblib.load(scaler_path)

    df_single = pd.DataFrame([patient_data])
    for col in FEATURE_COLS:
        if col not in df_single.columns:
            raise KeyError(f"Missing required patient feature: '{col}'")

    df_proc = df_single[FEATURE_COLS].copy()
    
    # 1. Sex encoding ('m'/'f' or 1/0)
    sex_val = df_proc["Sex"].iloc[0]
    if isinstance(sex_val, str):
        df_proc["Sex"] = df_proc["Sex"].map(sex_encoder).astype(float)
    else:
        df_proc["Sex"] = float(sex_val)

    # 2. Imputation
    X_imp = imputer.transform(df_proc)
    
    # 3. Standardization
    X_scaled = scaler.transform(X_imp)

    if model_type.lower() == "ml":
        model = joblib.load(ml_model_path)
        pred_cls = int(model.predict(X_scaled)[0])
        label_str = CLASS_MAPPING[pred_cls]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(probs[pred_cls])
            prob_dict = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
        else:
            confidence = 1.0
            prob_dict = {label_str: 1.0}

    elif model_type.lower() == "qml":
        import torch
        import torch.nn as nn
        import pennylane as qml

        pca_path = find_artifact_file("hcv_pca_4.joblib", "preprocessing")
        weights_path = find_artifact_file("hcv_qml_model_weights.pt", "models")
        pca_4 = joblib.load(pca_path)

        class HCVQuantumClassifier(nn.Module):
            def __init__(self, n_qubits=4, n_layers=2, n_classes=2):
                super().__init__()
                self.n_qubits = n_qubits
                dev = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits))
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

                self.circuit = circuit
                weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
                self.weights = nn.Parameter(0.1 * torch.randn(weight_shape, dtype=torch.float32))
                self.readout = nn.Linear(n_qubits, n_classes)

            def forward(self, x):
                res = self.circuit(x, self.weights)
                q_out = torch.stack(res, dim=1).float()
                logits = self.readout(q_out)
                return logits

        q_model = HCVQuantumClassifier(n_qubits=4)
        q_model.load_state_dict(torch.load(weights_path))
        q_model.eval()

        X_pca = pca_4.transform(X_scaled)
        with torch.no_grad():
            logits = q_model(torch.tensor(X_pca, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_cls = int(torch.argmax(logits, dim=1)[0])

        label_str = CLASS_MAPPING[pred_cls]
        confidence = float(probs[pred_cls])
        prob_dict = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ml' or 'qml'.")

    return {
        "model_type": model_type.upper(),
        "prediction_class": pred_cls,
        "prediction_label": label_str,
        "confidence": confidence,
        "class_probabilities": prob_dict
    }


def main():
    parser = argparse.ArgumentParser(description="HCV Pathology Clinical Prediction Tool")
    parser.add_argument("--model", type=str, default="ml", choices=["ml", "qml"], help="Model type ('ml' or 'qml')")
    parser.add_argument("--Age", type=float, default=45.0, help="Age in years")
    parser.add_argument("--Sex", type=str, default="m", help="Sex ('m' or 'f')")
    parser.add_argument("--ALB", type=float, default=38.5, help="Albumin (g/L)")
    parser.add_argument("--ALP", type=float, default=52.5, help="Alkaline Phosphatase (U/L)")
    parser.add_argument("--ALT", type=float, default=25.0, help="Alanine Transaminase (U/L)")
    parser.add_argument("--AST", type=float, default=22.0, help="Aspartate Transaminase (U/L)")
    parser.add_argument("--BIL", type=float, default=11.0, help="Bilirubin (umol/L)")
    parser.add_argument("--CHE", type=float, default=8.5, help="Cholinesterase (kU/L)")
    parser.add_argument("--CHOL", type=float, default=5.2, help="Cholesterol (mmol/L)")
    parser.add_argument("--CREA", type=float, default=75.0, help="Creatinine (umol/L)")
    parser.add_argument("--GGT", type=float, default=18.0, help="Gamma-Glutamyl Transferase (U/L)")
    parser.add_argument("--PROT", type=float, default=72.0, help="Total Protein (g/L)")

    args = parser.parse_args()
    patient_input = {col: getattr(args, col) for col in FEATURE_COLS}

    print("=========================================================")
    print(f"   PATIENT CLINICAL INFERENCE ({args.model.upper()} ENGINE)    ")
    print("=========================================================")
    print("Input Clinical Biomarkers:")
    for k, v in patient_input.items():
        print(f"  - {k:8s}: {v}")

    result = predict_patient_hcv(patient_input, model_type=args.model)

    print("\n---------------------------------------------------------")
    print(f"Model Engine      : {result['model_type']}")
    print(f"HCV Prediction    : {result['prediction_label'].upper()}")
    print(f"Model Confidence  : {result['confidence']*100:.2f}%")
    print(f"Probabilities     : {result['class_probabilities']}")
    print("---------------------------------------------------------\n")


if __name__ == "__main__":
    main()
