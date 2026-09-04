"""
Audited Heart Disease Inference Engine (Module: HEART/src/inference/predict_heart.py)
-----------------------------------------------------------------------------------
Accepts patient clinical parameters, validates ranges, transforms with saved preprocessors,
and predicts presence or absence of significant Coronary Artery Disease (CAD) using
the top classical ML model (HistGradientBoosting) or PennyLane 4-qubit Quantum VQC.
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
    0: "Absence of Significant CAD (<50% Stenosis)",
    1: "Presence of Significant CAD (>=50% Stenosis)"
}

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_RANGES = {
    "age": (18.0, 100.0, "Age in years"),
    "sex": (0.0, 1.0, "Sex (1=male, 0=female)"),
    "cp": (1.0, 4.0, "Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)"),
    "trestbps": (70.0, 250.0, "Resting BP (mm Hg)"),
    "chol": (100.0, 600.0, "Serum cholesterol (mg/dl)"),
    "fbs": (0.0, 1.0, "Fasting blood sugar > 120 (1=true, 0=false)"),
    "restecg": (0.0, 2.0, "Resting ECG (0=normal, 1=ST-T abnormality, 2=LVH)"),
    "thalach": (60.0, 230.0, "Max heart rate (bpm)"),
    "exang": (0.0, 1.0, "Exercise induced angina (1=yes, 0=no)"),
    "oldpeak": (0.0, 10.0, "ST depression (mm)"),
    "slope": (1.0, 3.0, "ST slope (1=upsloping, 2=flat, 3=downsloping)"),
    "ca": (0.0, 3.0, "Major vessels (0-3)"),
    "thal": (3.0, 7.0, "Thal defect (3=normal, 6=fixed, 7=reversible)")
}


def find_artifact_file(filename, subfolder=None):
    if subfolder:
        sub_path = os.path.join(ARTIFACTS_DIR, subfolder, filename)
        if os.path.exists(sub_path):
            return sub_path
    root_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    raise FileNotFoundError(f"Artifact '{filename}' not found in {ARTIFACTS_DIR} (subfolder: {subfolder})")


def validate_patient_data(patient_data: dict):
    for col in FEATURE_COLS:
        if col not in patient_data:
            raise KeyError(f"Missing mandatory feature: '{col}'")
        val = float(patient_data[col])
        min_v, max_v, desc = FEATURE_RANGES[col]
        if val < min_v or val > max_v:
            raise ValueError(f"Feature '{col}' value {val} outside permissible physiological range [{min_v}, {max_v}] ({desc})")


def predict_patient_heart(patient_data: dict, model_type: str = "ml") -> dict:
    """
    Predict CAD status for a single patient dictionary.
    
    Parameters:
        patient_data (dict): Dictionary containing all 13 clinical biomarkers.
        model_type (str): 'ml' for top Classical ML model, 'qml' for 4-qubit Quantum Classifier.
    """
    validate_patient_data(patient_data)

    imputer_path = find_artifact_file("heart_imputer.joblib", "preprocessing")
    scaler_path = find_artifact_file("heart_scaler.joblib", "preprocessing")
    ml_model_path = find_artifact_file("best_heart_ml_model.joblib", "models")

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    df_single = pd.DataFrame([patient_data])[FEATURE_COLS].astype(float)
    X_imp = imputer.transform(df_single)
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

        model_desc = f"Classical ML ({model.__class__.__name__})"

    elif model_type.lower() == "qml":
        import torch
        import torch.nn as nn
        import pennylane as qml

        pca_path = find_artifact_file("heart_pca_4.joblib", "preprocessing")
        weights_path = find_artifact_file("heart_qml_model_weights.pt", "models")
        pca_4 = joblib.load(pca_path)

        class HeartQuantumClassifier(nn.Module):
            def __init__(self, n_qubits=4, n_layers=2, n_classes=2):
                super().__init__()
                self.n_qubits = n_qubits
                self.n_layers = n_layers
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
                return self.readout(q_out)

        q_model = HeartQuantumClassifier(n_qubits=4)
        q_model.load_state_dict(torch.load(weights_path, weights_only=True))
        q_model.eval()

        X_pca = pca_4.transform(X_scaled)
        with torch.no_grad():
            logits = q_model(torch.tensor(X_pca, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_cls = int(torch.argmax(logits, dim=1)[0].item())

        label_str = CLASS_MAPPING[pred_cls]
        confidence = float(probs[pred_cls])
        prob_dict = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
        model_desc = "PennyLane PyTorch VQC (4 Qubits)"

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'ml' or 'qml'.")

    # Generate clinical interpretability explanation
    key_drivers = []
    if float(patient_data["thal"]) == 7.0:
        key_drivers.append("Reversible thallium defect indicates stress-induced myocardial ischemia")
    elif float(patient_data["thal"]) == 6.0:
        key_drivers.append("Fixed thallium defect indicates non-viable scar tissue / prior infarction")

    if float(patient_data["ca"]) >= 1.0:
        key_drivers.append(f"{int(patient_data['ca'])} major coronary vessel(s) opacified by fluoroscopy")

    if float(patient_data["oldpeak"]) >= 1.5:
        key_drivers.append(f"Significant exercise-induced ST depression ({patient_data['oldpeak']} mm)")

    if float(patient_data["exang"]) == 1.0:
        key_drivers.append("Presence of exercise-induced angina")

    if float(patient_data["cp"]) == 4.0:
        key_drivers.append("Asymptomatic presentation (silent myocardial ischemia pattern)")

    if not key_drivers:
        key_drivers.append("All primary cardiological markers (thal, ca, ST depression) lie within low-risk limits")

    explanation = "; ".join(key_drivers)

    return {
        "model_used": model_desc,
        "prediction_class": pred_cls,
        "prediction_label": label_str,
        "confidence": confidence,
        "class_probabilities": prob_dict,
        "clinical_rationale": explanation
    }


def main():
    parser = argparse.ArgumentParser(description="Clinical Inference Tool for Coronary Artery Disease (CAD)")
    parser.add_argument("--age", type=float, required=True, help="Age in years (29-77)")
    parser.add_argument("--sex", type=float, required=True, help="Sex (1=male, 0=female)")
    parser.add_argument("--cp", type=float, required=True, help="Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)")
    parser.add_argument("--trestbps", type=float, required=True, help="Resting blood pressure in mm Hg (94-200)")
    parser.add_argument("--chol", type=float, required=True, help="Serum cholesterol in mg/dl (126-564)")
    parser.add_argument("--fbs", type=float, required=True, help="Fasting blood sugar > 120 (1=true, 0=false)")
    parser.add_argument("--restecg", type=float, required=True, help="Resting ECG (0=normal, 1=ST-T abnormality, 2=LVH)")
    parser.add_argument("--thalach", type=float, required=True, help="Max heart rate achieved (71-202 bpm)")
    parser.add_argument("--exang", type=float, required=True, help="Exercise induced angina (1=yes, 0=no)")
    parser.add_argument("--oldpeak", type=float, required=True, help="ST depression induced by exercise (0.0-6.2 mm)")
    parser.add_argument("--slope", type=float, required=True, help="Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)")
    parser.add_argument("--ca", type=float, required=True, help="Number of major vessels colored by fluoroscopy (0-3)")
    parser.add_argument("--thal", type=float, required=True, help="Thallium stress test (3=normal, 6=fixed defect, 7=reversible defect)")
    parser.add_argument("--model", type=str, default="ml", choices=["ml", "qml"], help="Inference model: 'ml' or 'qml'")

    args = parser.parse_args()
    patient = {
        "age": args.age, "sex": args.sex, "cp": args.cp, "trestbps": args.trestbps,
        "chol": args.chol, "fbs": args.fbs, "restecg": args.restecg, "thalach": args.thalach,
        "exang": args.exang, "oldpeak": args.oldpeak, "slope": args.slope, "ca": args.ca,
        "thal": args.thal
    }

    result = predict_patient_heart(patient, model_type=args.model)

    print("\n" + "=" * 65)
    print("      HEART DISEASE (CAD) CLINICAL INFERENCE REPORT")
    print("=" * 65)
    print(f"Model Engine     : {result['model_used']}")
    print(f"Diagnostic Result: {result['prediction_label']} (Class {result['prediction_class']})")
    print(f"Confidence Score : {result['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    for cls_name, prob in result["class_probabilities"].items():
        print(f"  - {cls_name}: {prob*100:.2f}%")
    print(f"\nClinical Rationale:\n  {result['clinical_rationale']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
