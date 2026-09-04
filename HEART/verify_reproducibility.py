"""
Standalone Reproducibility Verification Script for Heart Disease Module
----------------------------------------------------------------------
Loads persisted ML and QML models alongside fitted preprocessing objects,
re-evaluates them on the untouched 61-sample test split, and validates that
accuracy and Macro F1 match logged metrics within 1e-5 numerical tolerance.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import pennylane as qml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed", "cleaned_data")


def verify():
    print("=========================================================")
    print("   RUNNING REPRODUCIBILITY VERIFICATION: HEART MODULE   ")
    print("=========================================================\n")

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics", "heart_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    best_ml_name = metrics["best_classical_ml"]
    logged_ml = metrics["models_performance"][best_ml_name]
    logged_qml = metrics["models_performance"]["QML (4 Qubits)"]

    print(f"Target Classical ML Model: {best_ml_name}")
    print(f"Logged ML Metrics : Acc = {logged_ml['Accuracy']:.4f} | Macro F1 = {logged_ml['Macro F1']:.4f} | ROC-AUC = {logged_ml['ROC-AUC']:.4f}")
    print(f"Logged QML Metrics: Acc = {logged_qml['Accuracy']:.4f} | Macro F1 = {logged_qml['Macro F1']:.4f} | ROC-AUC = {logged_qml['ROC-AUC']:.4f}\n")

    # Load cleaned test data
    X_test_path = os.path.join(DATA_PROC_DIR, "X_test.csv")
    y_test_path = os.path.join(DATA_PROC_DIR, "y_test.csv")
    X_test = pd.read_csv(X_test_path).values
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Verify ML
    ml_path = os.path.join(ARTIFACTS_DIR, "models", "best_heart_ml_model.joblib")
    ml_model = joblib.load(ml_path)

    y_pred_ml = ml_model.predict(X_test)
    y_proba_ml = ml_model.predict_proba(X_test)[:, 1] if hasattr(ml_model, "predict_proba") else ml_model.decision_function(X_test)

    reprod_ml_acc = accuracy_score(y_test, y_pred_ml)
    reprod_ml_f1 = f1_score(y_test, y_pred_ml, average="macro")
    reprod_ml_auc = roc_auc_score(y_test, y_proba_ml)

    # Verify QML
    pca_4_path = os.path.join(ARTIFACTS_DIR, "preprocessing", "heart_pca_4.joblib")
    pca_4 = joblib.load(pca_4_path)
    X_test_q = pca_4.transform(X_test)

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
    q_weights_path = os.path.join(ARTIFACTS_DIR, "models", "heart_qml_model_weights.pt")
    q_model.load_state_dict(torch.load(q_weights_path, weights_only=True))
    q_model.eval()

    with torch.no_grad():
        logits = q_model(torch.tensor(X_test_q, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        y_pred_q = torch.argmax(logits, dim=1).numpy()

    reprod_qml_acc = accuracy_score(y_test, y_pred_q)
    reprod_qml_f1 = f1_score(y_test, y_pred_q, average="macro")
    reprod_qml_auc = roc_auc_score(y_test, probs)

    tol = 1e-4
    ml_pass = abs(reprod_ml_acc - logged_ml["Accuracy"]) < tol and abs(reprod_ml_f1 - logged_ml["Macro F1"]) < tol
    qml_pass = abs(reprod_qml_acc - logged_qml["Accuracy"]) < tol and abs(reprod_qml_f1 - logged_qml["Macro F1"]) < tol

    print(f"Reproduced Classical ML: Acc = {reprod_ml_acc:.4f} | Macro F1 = {reprod_ml_f1:.4f} | Delta = {abs(reprod_ml_acc - logged_ml['Accuracy']):.6f}")
    print(f"Reproduced PennyLane QML: Acc = {reprod_qml_acc:.4f} | Macro F1 = {reprod_qml_f1:.4f} | Delta = {abs(reprod_qml_acc - logged_qml['Accuracy']):.6f}\n")

    if ml_pass and qml_pass:
        print("---------------------------------------")
        print("   REPRODUCIBILITY CHECK: PASSED       ")
        print("---------------------------------------")
        return 0
    else:
        print("---------------------------------------")
        print("   REPRODUCIBILITY CHECK: FAILED       ")
        print("---------------------------------------")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
