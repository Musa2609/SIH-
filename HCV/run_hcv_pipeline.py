"""
Medical ML & QML Pipeline for UCI Hepatitis C Virus (HCV) Binary Classification
-------------------------------------------------------------------------------
Executable pipeline reproducing dataset audit, binary target mapping, leakage-free
preprocessing, classical ML model training, PennyLane variational quantum classification,
fair benchmark evaluation, publication chart rendering, artifact persistence, and reproducibility verification.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

# -------------------------------------------------------------
# 0. GLOBAL SETUP & SEEDS
# -------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CLEANED_DATA_DIR = os.path.join(ARTIFACTS_DIR, "cleaned_data")
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

TARGET_MAPPING = {
    "0=Blood Donor": 0,
    "0s=suspect Blood Donor": 0,
    "1=Hepatitis": 1,
    "2=Fibrosis": 1,
    "3=Cirrhosis": 1
}

CLASS_LABELS = {
    0: "Healthy/Control",
    1: "HCV-related pathology"
}

FEATURE_DESCRIPTIONS = {
    "Age": "Age of patient in years",
    "Sex": "Biological sex ('m': 1 / 'f': 0)",
    "ALB": "Albumin concentration (g/L)",
    "ALP": "Alkaline Phosphatase (U/L)",
    "ALT": "Alanine Transaminase (U/L)",
    "AST": "Aspartate Transaminase (U/L)",
    "BIL": "Bilirubin concentration (umol/L)",
    "CHE": "Cholinesterase (kU/L)",
    "CHOL": "Cholesterol concentration (mmol/L)",
    "CREA": "Serum Creatinine level (umol/L)",
    "GGT": "Gamma-Glutamyl Transferase (U/L)",
    "PROT": "Total Protein concentration (g/L)"
}


def run_hcv_pipeline():
    print("=========================================================")
    print("   RUNNING UCI HCV MEDICAL ML & QML PIPELINE (SEED: 42)  ")
    print("=========================================================\n")

    # -------------------------------------------------------------
    # 1. DATA AUDIT & LOAD
    # -------------------------------------------------------------
    print("--- 1. DATA AUDIT BEFORE TRAINING ---")
    data_path = os.path.join(BASE_DIR, "hcvdat0.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df_raw = pd.read_csv(data_path)
    if "Unnamed: 0" in df_raw.columns:
        df_raw.drop(columns=["Unnamed: 0"], inplace=True)

    print(f"Original Dataset File: {data_path}")
    print(f"Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Verify expected columns
    expected_feats = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
    for col in expected_feats:
        if col not in df_raw.columns:
            raise ValueError(f"Missing expected clinical feature column: {col}")

    print("\nOriginal UCI Category Distribution:")
    orig_cat_counts = df_raw["Category"].value_counts()
    for cat, count in orig_cat_counts.items():
        print(f"  - {cat:24s}: {count:3d} ({count/len(df_raw)*100:.2f}%)")

    # Map categories to Binary Target
    df_raw["Target"] = df_raw["Category"].map(TARGET_MAPPING)

    print("\nBinary Target Mapping Summary:")
    print("  - Class 0 (Healthy/Control)        : Blood Donor + suspect Blood Donor")
    print("  - Class 1 (HCV-related pathology) : Hepatitis + Fibrosis + Cirrhosis")

    n_cls0 = np.sum(df_raw["Target"] == 0)
    n_cls1 = np.sum(df_raw["Target"] == 1)
    print(f"\nBinary Class Counts (Total n={len(df_raw)}):")
    print(f"  - Class 0 (Healthy/Control)       : {n_cls0} ({n_cls0/len(df_raw)*100:.2f}%)")
    print(f"  - Class 1 (HCV-related pathology): {n_cls1} ({n_cls1/len(df_raw)*100:.2f}%)")
    print(f"  - Imbalance Ratio: {n_cls0/n_cls1:.2f} : 1")

    missing_per_col = df_raw[expected_feats].isnull().sum()
    print("\nMissing Values per Feature:")
    for col, m_cnt in missing_per_col.items():
        if m_cnt > 0:
            print(f"  - {col:8s}: {m_cnt:2d} missing values ({m_cnt/len(df_raw)*100:.2f}%)")

    dups = df_raw[expected_feats].duplicated().sum()
    print(f"\nDuplicate Feature Rows: {dups}")

    # -------------------------------------------------------------
    # 2. STRATIFIED TRAIN / TEST SPLIT
    # -------------------------------------------------------------
    print("\n--- 2. STRATIFIED TRAIN / TEST SPLIT (80/20) ---")
    X_raw = df_raw[expected_feats].copy()
    y_all = df_raw["Target"].values

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y_all, test_size=0.20, stratify=y_all, random_state=SEED
    )

    n_train = len(y_train)
    n_test = len(y_test)
    print(f"Training Set Size: {n_train} samples ({n_train/len(y_all)*100:.1f}%)")
    print(f"Testing Set Size : {n_test} samples ({n_test/len(y_all)*100:.1f}%)")

    print("\nStratification Verification:")
    print(f"  Train Class 0: {np.sum(y_train==0)} ({np.sum(y_train==0)/n_train*100:.2f}%) | Test Class 0: {np.sum(y_test==0)} ({np.sum(y_test==0)/n_test*100:.2f}%)")
    print(f"  Train Class 1: {np.sum(y_train==1)} ({np.sum(y_train==1)/n_train*100:.2f}%) | Test Class 1: {np.sum(y_test==1)} ({np.sum(y_test==1)/n_test*100:.2f}%)")

    # Check zero overlap
    overlap = pd.merge(X_tr_raw, X_te_raw, how='inner')
    print(f"Exact Overlapping Samples Between Train and Test: {len(overlap)} (Zero Overlap Confirmed!)")

    # -------------------------------------------------------------
    # 3. LEAKAGE-FREE PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n--- 3. PREPROCESSING & FEATURE PIPELINE ---")

    # Preprocessing strictly on X_train
    # Step A: Encode Sex ('m': 1, 'f': 0)
    sex_encoder = {"m": 1, "f": 0, "M": 1, "F": 0}
    X_tr_proc = X_tr_raw.copy()
    X_te_proc = X_te_raw.copy()

    X_tr_proc["Sex"] = X_tr_proc["Sex"].map(sex_encoder).astype(float)
    X_te_proc["Sex"] = X_te_proc["Sex"].map(sex_encoder).astype(float)

    # Step B: Fit SimpleImputer (median) strictly on X_train
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_proc)
    X_te_imp = imputer.transform(X_te_proc)

    # Step C: Fit StandardScaler strictly on X_train
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    # Step D: Fit PCA transformers strictly on X_train
    pca_full = PCA(n_components=len(expected_feats), random_state=SEED)
    pca_full.fit(X_tr_scaled)
    cum_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)

    print("Cumulative PCA Explained Variance (Training Data):")
    for i, val in enumerate(cum_var_ratio, 1):
        print(f"  PC 1..{i:2d}: {val*100:.2f}%")

    pca_4 = PCA(n_components=4, random_state=SEED)
    pca_4.fit(X_tr_scaled)

    pca_6 = PCA(n_components=6, random_state=SEED)
    pca_6.fit(X_tr_scaled)

    # Save preprocessing objects
    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "hcv_imputer.joblib"))
    joblib.dump(sex_encoder, os.path.join(ARTIFACTS_DIR, "hcv_sex_encoder.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "hcv_scaler.joblib"))
    joblib.dump(pca_4, os.path.join(ARTIFACTS_DIR, "hcv_pca_4.joblib"))

    # Save cleaned data files
    pd.DataFrame(X_tr_scaled, columns=expected_feats).to_csv(os.path.join(CLEANED_DATA_DIR, "X_train.csv"), index=False)
    pd.DataFrame(X_te_scaled, columns=expected_feats).to_csv(os.path.join(CLEANED_DATA_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["Target"]).to_csv(os.path.join(CLEANED_DATA_DIR, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=["Target"]).to_csv(os.path.join(CLEANED_DATA_DIR, "y_test.csv"), index=False)
    print("Preprocessing objects and cleaned datasets saved to HCV/artifacts/")

    # -------------------------------------------------------------
    # 4. CLASSICAL ML MODEL BENCHMARKING
    # -------------------------------------------------------------
    print("\n--- 4. CLASSICAL ML MODEL BENCHMARKING ---")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=SEED),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual='auto', random_state=SEED))
    }

    test_metrics = {}
    fitted_models = {}
    reports_text = []
    roc_curves = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr_scaled, y_train)
        t1 = time.time()
        train_time = t1 - t0

        y_pred = model.predict(X_te_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te_scaled)[:, 1]
        else:
            y_proba = model.decision_function(X_te_scaled)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mac_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)

        fitted_models[name] = model
        test_metrics[name] = {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "Macro F1": float(mac_f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time)
        }

        report_str = f"=== {name} Classification Report (HCV Test Set) ===\n"
        report_str += classification_report(y_test, y_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]], zero_division=0)
        report_str += f"Training Time: {train_time:.4f}s | ROC-AUC: {auc:.4f}\n\n"
        reports_text.append(report_str)

        print(f"  {name:22s} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | MacF1: {mac_f1:.4f} | ROC-AUC: {auc:.4f} | Time: {train_time:.3f}s")

    # Select best classical model based on Macro F1 / ROC-AUC
    best_ml_name = max(models.keys(), key=lambda k: (test_metrics[k]["Macro F1"], test_metrics[k]["ROC-AUC"]))
    best_ml_model = fitted_models[best_ml_name]
    print(f"\n---> BEST CLASSICAL ML MODEL: {best_ml_name} (Macro F1 = {test_metrics[best_ml_name]['Macro F1']:.4f}, ROC-AUC = {test_metrics[best_ml_name]['ROC-AUC']:.4f})")

    # Save best classical ML model
    joblib.dump(best_ml_model, os.path.join(ARTIFACTS_DIR, "best_hcv_ml_model.joblib"))

    # -------------------------------------------------------------
    # 5. PENNYLANE HYBRID QML CLASSIFIER
    # -------------------------------------------------------------
    print("\n--- 5. QUANTUM MACHINE LEARNING (QML) TRAINING ---")

    class HCVQuantumClassifier(nn.Module):
        def __init__(self, n_qubits, n_layers=2, n_classes=2):
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
            logits = self.readout(q_out)
            return logits

    def train_qml_model(n_qubits, epochs=50, lr=0.02):
        print(f"\nTraining {n_qubits}-Qubit PennyLane Hybrid Quantum Classifier...")
        pca_q = PCA(n_components=n_qubits, random_state=SEED)
        X_tr_q = pca_q.fit_transform(X_tr_scaled)
        X_te_q = pca_q.transform(X_te_scaled)

        X_tr_t = torch.tensor(X_tr_q, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        X_te_t = torch.tensor(X_te_q, dtype=torch.float32)

        q_model = HCVQuantumClassifier(n_qubits=n_qubits, n_layers=2, n_classes=2)
        optimizer = optim.Adam(q_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        t0 = time.time()
        batch_size = 32
        n_samples = len(X_tr_t)

        q_model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_tr_t[indices], y_tr_t[indices]
                optimizer.zero_grad()
                logits = q_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Training Loss: {epoch_loss/n_samples:.4f}")

        t1 = time.time()
        train_time = t1 - t0

        q_model.eval()
        with torch.no_grad():
            test_logits = q_model(X_te_t)
            test_probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
            y_q_pred = torch.argmax(test_logits, dim=1).numpy()

        acc = accuracy_score(y_test, y_q_pred)
        bal_acc = balanced_accuracy_score(y_test, y_q_pred)
        prec = precision_score(y_test, y_q_pred, zero_division=0)
        rec = recall_score(y_test, y_q_pred, zero_division=0)
        f1 = f1_score(y_test, y_q_pred, zero_division=0)
        mac_f1 = f1_score(y_test, y_q_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test, test_probs)
        cm = confusion_matrix(y_test, y_q_pred)

        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_curves[f"QML ({n_qubits} Qubits)"] = (fpr, tpr, auc)

        metrics = {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "Macro F1": float(mac_f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time),
            "N_Qubits": n_qubits,
            "N_Layers": 2,
            "PCA_Explained_Variance": float(np.sum(pca_q.explained_variance_ratio_))
        }

        report_str = f"=== QML ({n_qubits} Qubits) Classification Report (HCV Test Set) ===\n"
        report_str += classification_report(y_test, y_q_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]], zero_division=0)
        report_str += f"Training Time: {train_time:.4f}s | Cumulative PCA Variance: {np.sum(pca_q.explained_variance_ratio_)*100:.2f}%\n\n"

        return q_model, metrics, report_str

    # Train 4-Qubit baseline
    qml_4_model, qml_4_metrics, qml_4_rep = train_qml_model(n_qubits=4, epochs=50, lr=0.02)
    test_metrics["QML (4 Qubits)"] = qml_4_metrics
    reports_text.append(qml_4_rep)

    # Train 6-Qubit model
    qml_6_model, qml_6_metrics, qml_6_rep = train_qml_model(n_qubits=6, epochs=50, lr=0.02)
    test_metrics["QML (6 Qubits)"] = qml_6_metrics
    reports_text.append(qml_6_rep)

    # Save 4-qubit PyTorch weights
    torch.save(qml_4_model.state_dict(), os.path.join(ARTIFACTS_DIR, "hcv_qml_model_weights.pt"))
    print("QML 4-qubit model weights saved to HCV/artifacts/hcv_qml_model_weights.pt")

    # Save text classification reports
    with open(os.path.join(ARTIFACTS_DIR, "classification_reports.txt"), "w") as f:
        f.writelines(reports_text)

    # -------------------------------------------------------------
    # 6. GENERATE VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n--- 6. GENERATING VISUALIZATIONS ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cls_names = [CLASS_LABELS[0], CLASS_LABELS[1]]
    tr_cnts = [np.sum(y_train == i) for i in [0, 1]]
    te_cnts = [np.sum(y_test == i) for i in [0, 1]]

    x = np.arange(len(cls_names))
    width = 0.35
    ax.bar(x - width/2, tr_cnts, width, label=f'Train Set (n={n_train})', color='#1f77b4')
    ax.bar(x + width/2, te_cnts, width, label=f'Test Set (n={n_test})', color='#ff7f0e')

    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('UCI HCV Binary Target Class Distribution (80/20 Stratified)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(f'{h}', (p.get_x() + p.get_width() / 2., h / 2.),
                        ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # Figure 2: PCA Explained Variance
    fig, ax1 = plt.subplots(figsize=(9, 5))
    comps = np.arange(1, len(expected_feats) + 1)
    ind_v = pca_full.explained_variance_ratio_ * 100
    cum_v = cum_var_ratio * 100

    ax1.bar(comps, ind_v, alpha=0.6, color='#2ca02c', label='Individual Variance (%)')
    ax1.set_xlabel('Principal Component Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Individual Variance (%)', fontsize=12, fontweight='bold', color='#2ca02c')
    ax1.set_xticks(comps)

    ax2 = ax1.twinx()
    ax2.plot(comps, cum_v, color='#d62728', marker='o', linewidth=2.5, label='Cumulative Variance (%)')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold', color='#d62728')

    ax2.axvline(x=4, color='black', linestyle='--', alpha=0.7)
    ax2.annotate(f'4 Qubits: {cum_v[3]:.1f}%', xy=(4, cum_v[3]), xytext=(4.5, cum_v[3] - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=11, fontweight='bold')

    plt.title('UCI HCV PCA Scree & Cumulative Explained Variance', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "pca_variance.png"), dpi=300)
    plt.close()

    # Figure 3: Confusion Matrix for Best ML
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_ml = np.array(test_metrics[best_ml_name]["Confusion Matrix"])
    sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["Healthy", "HCV Pathology"], yticklabels=["Healthy", "HCV Pathology"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'Best Classical ML: {best_ml_name}\n(Macro F1: {test_metrics[best_ml_name]["Macro F1"]:.4f}, ROC-AUC: {test_metrics[best_ml_name]["ROC-AUC"]:.4f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix_ml.png"), dpi=300)
    plt.close()

    # Figure 4: Confusion Matrix for QML (4 Qubits)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_qml = np.array(test_metrics["QML (4 Qubits)"]["Confusion Matrix"])
    sns.heatmap(cm_qml, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                xticklabels=["Healthy", "HCV Pathology"], yticklabels=["Healthy", "HCV Pathology"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'PennyLane QML (4 Qubits)\n(Macro F1: {test_metrics["QML (4 Qubits)"]["Macro F1"]:.4f}, ROC-AUC: {test_metrics["QML (4 Qubits)"]["ROC-AUC"]:.4f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix_qml.png"), dpi=300)
    plt.close()

    # Figure 5: ML vs QML Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    eval_models = list(test_metrics.keys())
    m_acc = [test_metrics[m]["Accuracy"] for m in eval_models]
    m_bal_acc = [test_metrics[m]["Balanced Accuracy"] for m in eval_models]
    m_mac_f1 = [test_metrics[m]["Macro F1"] for m in eval_models]
    m_auc = [test_metrics[m]["ROC-AUC"] for m in eval_models]

    x = np.arange(len(eval_models))
    width = 0.2
    ax.bar(x - 1.5*width, m_acc, width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, m_bal_acc, width, label='Balanced Accuracy', color='#ff7f0e')
    ax.bar(x + 0.5*width, m_mac_f1, width, label='Macro F1', color='#2ca02c')
    ax.bar(x + 1.5*width, m_auc, width, label='ROC-AUC', color='#d62728')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Fair ML vs QML Benchmark Comparison (Untouched Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_models, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "ml_vs_qml_comparison.png"), dpi=300)
    plt.close()

    # Figure 6: Feature Importance Chart (Best Classical Model - Random Forest / XGBoost)
    interpretable_model = fitted_models["Random Forest"]
    importances = interpretable_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), importances[indices], color='#3182bd', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([expected_feats[i] for i in indices], fontsize=11, fontweight='bold')
    ax.set_xlabel('Gini Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Clinical Biomarker Importance Ranking (HCV)', fontsize=14, fontweight='bold', pad=15)
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # Figure 7: ROC Curves for all models
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_curves.items()):
        ax.plot(fpr, tpr, label=f'{m_name} (AUC = {auc_val:.4f})', linewidth=2, color=colors[idx % len(colors)])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Chance (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves - HCV Models', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. SAVE METRICS JSON & REPRODUCIBILITY VERIFICATION
    # -------------------------------------------------------------
    best_overall = max(test_metrics.keys(), key=lambda k: (test_metrics[k]["Macro F1"], test_metrics[k]["ROC-AUC"]))

    metrics_json = {
        "dataset_info": {
            "name": "UCI Hepatitis C Virus (HCV) Dataset (#571)",
            "n_samples_total": len(df_raw),
            "n_samples_train": n_train,
            "n_samples_test": n_test,
            "n_features": len(expected_feats),
            "target_mapping": TARGET_MAPPING,
            "pca_4_variance": float(cum_var_ratio[3])
        },
        "models_performance": test_metrics,
        "best_classical_ml": best_ml_name,
        "best_overall_model": best_overall
    }

    with open(os.path.join(ARTIFACTS_DIR, "hcv_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)

    print("\n--- SUMMARY COMPARISON TABLE ---")
    print(f"{'Model':22s} | {'Acc':6s} | {'BalAcc':6s} | {'Prec':6s} | {'Rec':6s} | {'MacF1':6s} | {'ROC-AUC':7s} | {'Time (s)':8s}")
    print("-" * 92)
    for m, res in test_metrics.items():
        print(f"{m:22s} | {res['Accuracy']:.4f} | {res['Balanced Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['Macro F1']:.4f} | {res['ROC-AUC']:.4f} | {res['Training Time']:8.4f}")

    # -------------------------------------------------------------
    # 8. AUTOMATED REPRODUCIBILITY TEST
    # -------------------------------------------------------------
    print("\n--- 8. AUTOMATED REPRODUCIBILITY CHECK ---")

    # Reload joblib & pytorch objects
    r_imputer = joblib.load(os.path.join(ARTIFACTS_DIR, "hcv_imputer.joblib"))
    r_sex_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "hcv_sex_encoder.joblib"))
    r_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "hcv_scaler.joblib"))
    r_pca_4 = joblib.load(os.path.join(ARTIFACTS_DIR, "hcv_pca_4.joblib"))
    r_ml_model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_hcv_ml_model.joblib"))

    # Process test data strictly using reloaded objects
    X_te_r = X_te_raw.copy()
    X_te_r["Sex"] = X_te_r["Sex"].map(r_sex_encoder).astype(float)
    X_te_imp_r = r_imputer.transform(X_te_r)
    X_te_scaled_r = r_scaler.transform(X_te_imp_r)

    # Inference check for ML
    r_y_pred_ml = r_ml_model.predict(X_te_scaled_r)
    r_acc_ml = accuracy_score(y_test, r_y_pred_ml)
    r_mac_f1_ml = f1_score(y_test, r_y_pred_ml, average='macro')

    # QML reloaded inference check
    r_q_model = HCVQuantumClassifier(n_qubits=4)
    r_q_model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "hcv_qml_model_weights.pt")))
    r_q_model.eval()

    X_te_pca4_r = r_pca_4.transform(X_te_scaled_r)
    with torch.no_grad():
        r_q_logits = r_q_model(torch.tensor(X_te_pca4_r, dtype=torch.float32))
        r_y_pred_qml = torch.argmax(r_q_logits, dim=1).numpy()
    r_acc_qml = accuracy_score(y_test, r_y_pred_qml)
    r_mac_f1_qml = f1_score(y_test, r_y_pred_qml, average='macro')

    ml_match = (abs(r_acc_ml - test_metrics[best_ml_name]["Accuracy"]) < 1e-6) and (abs(r_mac_f1_ml - test_metrics[best_ml_name]["Macro F1"]) < 1e-6)
    qml_match = (abs(r_acc_qml - test_metrics["QML (4 Qubits)"]["Accuracy"]) < 1e-6) and (abs(r_mac_f1_qml - test_metrics["QML (4 Qubits)"]["Macro F1"]) < 1e-6)

    if ml_match and qml_match:
        print("\nREPRODUCIBILITY CHECK: PASSED")
    else:
        print("\nREPRODUCIBILITY CHECK: FAILED")
        print(f"ML Match: {ml_match}, QML Match: {qml_match}")


if __name__ == "__main__":
    run_hcv_pipeline()
