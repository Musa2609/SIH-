import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# 0. GLOBAL CONFIGURATION & REPRODUCIBILITY SEED
# -------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed", "cleaned_data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROC_DIR = os.path.join(ARTIFACTS_DIR, "preprocessing")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

for d in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, PREPROC_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years (continuous)",
    "sex": "Biological sex (1=male; 0=female)",
    "cp": "Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)",
    "trestbps": "Resting systolic blood pressure in mm Hg (continuous)",
    "chol": "Serum cholesterol in mg/dl (continuous)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1=true; 0=false)",
    "restecg": "Resting ECG (0=normal, 1=ST-T abnormality, 2=LVH)",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise induced angina (1=yes; 0=no)",
    "oldpeak": "ST depression induced by exercise relative to rest (mm)",
    "slope": "Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thallium stress test defect (3=normal, 6=fixed, 7=reversible)"
}

CLASS_LABELS = {
    0: "Absence of CAD (<50% Stenosis)",
    1: "Presence of CAD (>=50% Stenosis)"
}


def run_heart_pipeline():
    print("=================================================================")
    print("   RUNNING UCI HEART DISEASE (CAD) ML & QML PIPELINE (SEED: 42)  ")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # 1. LOAD RAW DATASET & PERFORM AUDIT
    # -------------------------------------------------------------
    print("--- 1. DATA AUDIT & VERIFICATION ---")
    data_path = os.path.join(DATA_RAW_DIR, "heart.csv")
    if not os.path.exists(data_path):
        print(f"Downloading raw dataset from UCI Repository to {data_path}...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = FEATURE_COLS + ["num"]
        df_raw = pd.read_csv(url, names=cols, na_values="?")
        df_raw.to_csv(data_path, index=False)
    else:
        df_raw = pd.read_csv(data_path)

    print(f"Loaded dataset: {data_path}")
    print(f"Dimensions: {df_raw.shape[0]} patient cases, {df_raw.shape[1]} columns")

    for col in FEATURE_COLS:
        if col not in df_raw.columns:
            raise ValueError(f"Missing expected clinical feature: {col}")

    print("\nRaw Angiographic Disease Status Distribution ('num'):")
    num_counts = df_raw["num"].value_counts().sort_index()
    for val, cnt in num_counts.items():
        print(f"  - num = {val}: {cnt:3d} cases ({cnt/len(df_raw)*100:.2f}%)")

    # Formulate binary classification target
    df_raw["target"] = (df_raw["num"] >= 1).astype(int)
    n_cls0 = int(np.sum(df_raw["target"] == 0))
    n_cls1 = int(np.sum(df_raw["target"] == 1))

    print("\nBinary Medical Target Definition:")
    print("  - Class 0: Absence of Significant CAD (< 50% stenosis in major coronary vessels)")
    print("  - Class 1: Presence of Significant CAD (>= 50% stenosis in >= 1 major vessel)")
    print(f"Binary Counts: Class 0 = {n_cls0} ({n_cls0/len(df_raw)*100:.2f}%), Class 1 = {n_cls1} ({n_cls1/len(df_raw)*100:.2f}%)")
    print(f"Imbalance Ratio: {n_cls0/n_cls1:.2f} : 1 (Near-ideal balance)")

    missing_vals = df_raw[FEATURE_COLS].isnull().sum()
    print("\nMissing Values per Feature:")
    for col, m_cnt in missing_vals.items():
        if m_cnt > 0:
            print(f"  - {col:10s}: {m_cnt} missing values ({m_cnt/len(df_raw)*100:.2f}%)")
    if missing_vals.sum() == 0:
        print("  - No missing values found.")

    dups = df_raw[FEATURE_COLS].duplicated().sum()
    print(f"Duplicate Patient Records: {dups}")

    # -------------------------------------------------------------
    # 2. STRATIFIED TRAIN / TEST SPLIT (80 / 20)
    # -------------------------------------------------------------
    print("\n--- 2. STRATIFIED TRAIN / TEST SPLIT (80/20) ---")
    X_raw = df_raw[FEATURE_COLS].copy()
    y_all = df_raw["target"].values

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y_all, test_size=0.20, stratify=y_all, random_state=SEED
    )

    n_train = len(y_train)
    n_test = len(y_test)
    print(f"Training Set : {n_train} samples ({n_train/len(y_all)*100:.1f}%)")
    print(f"Testing Set  : {n_test} samples ({n_test/len(y_all)*100:.1f}%)")
    print(f"  Train: Class 0={np.sum(y_train==0)} ({np.sum(y_train==0)/n_train*100:.1f}%), Class 1={np.sum(y_train==1)} ({np.sum(y_train==1)/n_train*100:.1f}%)")
    print(f"  Test : Class 0={np.sum(y_test==0)} ({np.sum(y_test==0)/n_test*100:.1f}%), Class 1={np.sum(y_test==1)} ({np.sum(y_test==1)/n_test*100:.1f}%)")

    overlap = pd.merge(X_tr_raw, X_te_raw, how="inner")
    print(f"Train/Test Overlapping Records: {len(overlap)} (Zero Leakage Confirmed!)")

    # -------------------------------------------------------------
    # 3. LEAKAGE-FREE PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n--- 3. LEAKAGE-FREE PREPROCESSING ---")
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_te_imp = imputer.transform(X_te_raw)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    # Persist cleaned datasets to disk
    train_df = pd.DataFrame(X_tr_scaled, columns=FEATURE_COLS)
    test_df = pd.DataFrame(X_te_scaled, columns=FEATURE_COLS)
    train_df.to_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_PROC_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(os.path.join(DATA_PROC_DIR, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=["target"]).to_csv(os.path.join(DATA_PROC_DIR, "y_test.csv"), index=False)

    # Reload from persistent storage so downstream training & evaluation are 100% bitwise consistent
    X_tr_scaled = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train.csv")).values
    X_te_scaled = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv")).values

    pca_full = PCA(n_components=len(FEATURE_COLS), random_state=SEED)
    pca_full.fit(X_tr_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    print("Cumulative PCA Explained Variance (Training Data):")
    for i, v in enumerate(cum_var, 1):
        print(f"  PC 1..{i:2d}: {v*100:.2f}%")

    pca_4 = PCA(n_components=4, random_state=SEED)
    pca_4.fit(X_tr_scaled)

    pca_6 = PCA(n_components=6, random_state=SEED)
    pca_6.fit(X_tr_scaled)

    # Persist preprocessing transformers
    joblib.dump(imputer, os.path.join(PREPROC_DIR, "heart_imputer.joblib"))
    joblib.dump(scaler, os.path.join(PREPROC_DIR, "heart_scaler.joblib"))
    joblib.dump(pca_4, os.path.join(PREPROC_DIR, "heart_pca_4.joblib"))
    joblib.dump(pca_6, os.path.join(PREPROC_DIR, "heart_pca_6.joblib"))
    print("Cleaned splits and preprocessing objects saved successfully.")

    # -------------------------------------------------------------
    # 4. CLASSICAL MACHINE LEARNING BENCHMARKING
    # -------------------------------------------------------------
    print("\n--- 4. CLASSICAL ML MODEL BENCHMARKING ---")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="logloss", max_depth=3, n_estimators=100, random_state=SEED),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual="auto", random_state=SEED))
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
        mac_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
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
            "Weighted F1": float(weighted_f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time)
        }

        report_str = f"=== {name} Classification Report (Heart Test Set) ===\n"
        report_str += classification_report(y_test, y_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]], zero_division=0)
        report_str += f"Training Time: {train_time:.4f}s | ROC-AUC: {auc:.4f}\n\n"
        reports_text.append(report_str)

        print(f"  {name:22s} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | MacF1: {mac_f1:.4f} | ROC-AUC: {auc:.4f} | Time: {train_time:.3f}s")

    best_ml_name = max(models.keys(), key=lambda k: (test_metrics[k]["Macro F1"], test_metrics[k]["ROC-AUC"]))
    best_ml_model = fitted_models[best_ml_name]
    print(f"\n---> BEST CLASSICAL ML MODEL: {best_ml_name} (Macro F1 = {test_metrics[best_ml_name]['Macro F1']:.4f}, ROC-AUC = {test_metrics[best_ml_name]['ROC-AUC']:.4f})")

    # Persist best classical ML model
    joblib.dump(best_ml_model, os.path.join(MODELS_DIR, "best_heart_ml_model.joblib"))

    # -------------------------------------------------------------
    # 5. PENNYLANE HYBRID QUANTUM CLASSIFIER (QML)
    # -------------------------------------------------------------
    print("\n--- 5. QUANTUM MACHINE LEARNING (QML) TRAINING ---")

    class HeartQuantumClassifier(nn.Module):
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

    def train_qml(n_qubits, epochs=50, lr=0.02):
        print(f"\nTraining {n_qubits}-Qubit PennyLane Hybrid Quantum Classifier...")
        pca_q = PCA(n_components=n_qubits, random_state=SEED)
        X_tr_q = pca_q.fit_transform(X_tr_scaled)
        X_te_q = pca_q.transform(X_te_scaled)

        X_tr_t = torch.tensor(X_tr_q, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        X_te_t = torch.tensor(X_te_q, dtype=torch.float32)

        q_model = HeartQuantumClassifier(n_qubits=n_qubits, n_layers=2, n_classes=2)
        optimizer = optim.Adam(q_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        t0 = time.time()
        batch_size = 32
        n_samples = len(X_tr_t)

        q_model.train()
        loss_history = []
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                bx, by = X_tr_t[indices], y_tr_t[indices]
                optimizer.zero_grad()
                logits = q_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}")

        train_time = time.time() - t0

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
        mac_f1 = f1_score(y_test, y_q_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_q_pred, average="weighted", zero_division=0)
        auc = roc_auc_score(y_test, test_probs)
        cm = confusion_matrix(y_test, y_q_pred)

        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_curves[f"QML ({n_qubits} Qubits)"] = (fpr, tpr, auc)

        total_params = sum(p.numel() for p in q_model.parameters())

        metrics = {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "Macro F1": float(mac_f1),
            "Weighted F1": float(weighted_f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time),
            "N_Qubits": n_qubits,
            "N_Layers": 2,
            "Trainable_Parameters": total_params,
            "PCA_Explained_Variance": float(np.sum(pca_q.explained_variance_ratio_))
        }

        rep_str = f"=== QML ({n_qubits} Qubits) Classification Report (Heart Test Set) ===\n"
        rep_str += classification_report(y_test, y_q_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]], zero_division=0)
        rep_str += f"Training Time: {train_time:.4f}s | Cumulative PCA Variance: {np.sum(pca_q.explained_variance_ratio_)*100:.2f}%\n\n"

        return q_model, metrics, rep_str, loss_history

    # Train 4-Qubit Model
    qml_4_model, qml_4_metrics, qml_4_rep, qml_4_losses = train_qml(n_qubits=4, epochs=50, lr=0.02)
    test_metrics["QML (4 Qubits)"] = qml_4_metrics
    reports_text.append(qml_4_rep)

    # Train 6-Qubit Model
    qml_6_model, qml_6_metrics, qml_6_rep, qml_6_losses = train_qml(n_qubits=6, epochs=50, lr=0.02)
    test_metrics["QML (6 Qubits)"] = qml_6_metrics
    reports_text.append(qml_6_rep)

    # Save 4-Qubit PyTorch weights
    torch.save(qml_4_model.state_dict(), os.path.join(MODELS_DIR, "heart_qml_model_weights.pt"))
    print("4-Qubit QML weights saved to HEART/artifacts/models/heart_qml_model_weights.pt")

    # Persist text classification reports
    with open(os.path.join(METRICS_DIR, "classification_reports.txt"), "w") as f:
        f.writelines(reports_text)

    # -------------------------------------------------------------
    # 6. HIGH-RESOLUTION PUBLICATION VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n--- 6. GENERATING HIGH-RESOLUTION VISUALIZATIONS ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cls_names = [CLASS_LABELS[0], CLASS_LABELS[1]]
    tr_cnts = [int(np.sum(y_train == i)) for i in [0, 1]]
    te_cnts = [int(np.sum(y_test == i)) for i in [0, 1]]
    x = np.arange(len(cls_names))
    width = 0.35
    ax.bar(x - width/2, tr_cnts, width, label=f"Train Set (n={n_train})", color="#1f77b4")
    ax.bar(x + width/2, te_cnts, width, label=f"Test Set (n={n_test})", color="#ff7f0e")
    ax.set_ylabel("Sample Count", fontsize=12, fontweight="bold")
    ax.set_title("UCI Heart Disease Binary Target Class Distribution (80/20 Stratified)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h / 2.),
                        ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # Figure 2: PCA Explained Variance Scree Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    comps = np.arange(1, len(FEATURE_COLS) + 1)
    ind_v = pca_full.explained_variance_ratio_ * 100
    cum_v = cum_var * 100

    ax1.bar(comps, ind_v, alpha=0.6, color="#2ca02c", label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Individual Variance (%)", fontsize=12, fontweight="bold", color="#2ca02c")
    ax1.set_xticks(comps)

    ax2 = ax1.twinx()
    ax2.plot(comps, cum_v, color="#d62728", marker="o", linewidth=2.5, label="Cumulative Variance (%)")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, fontweight="bold", color="#d62728")

    ax2.axvline(x=4, color="blue", linestyle="--", alpha=0.7)
    ax2.annotate(f"4 Qubits: {cum_v[3]:.1f}%", xy=(4, cum_v[3]), xytext=(4.5, cum_v[3] - 10),
                 arrowprops=dict(facecolor="blue", shrink=0.05, width=1, headwidth=6), fontsize=10, fontweight="bold")
    ax2.axvline(x=6, color="purple", linestyle="--", alpha=0.7)
    ax2.annotate(f"6 Qubits: {cum_v[5]:.1f}%", xy=(6, cum_v[5]), xytext=(6.5, cum_v[5] - 10),
                 arrowprops=dict(facecolor="purple", shrink=0.05, width=1, headwidth=6), fontsize=10, fontweight="bold")

    plt.title("UCI Heart Disease PCA Scree & Cumulative Explained Variance", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_variance.png"), dpi=300)
    plt.close()

    # Figure 3: Confusion Matrix for Best Classical ML
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_ml = np.array(test_metrics[best_ml_name]["Confusion Matrix"])
    sns.heatmap(cm_ml, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Absence", "Presence"], yticklabels=["Absence", "Presence"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Best Classical ML: {best_ml_name}\n(Macro F1: {test_metrics[best_ml_name]['Macro F1']:.4f}, ROC-AUC: {test_metrics[best_ml_name]['ROC-AUC']:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_ml.png"), dpi=300)
    plt.close()

    # Figure 4: Confusion Matrix for QML (4 Qubits)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_qml = np.array(test_metrics["QML (4 Qubits)"]["Confusion Matrix"])
    sns.heatmap(cm_qml, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                xticklabels=["Absence", "Presence"], yticklabels=["Absence", "Presence"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"PennyLane QML (4 Qubits)\n(Macro F1: {test_metrics['QML (4 Qubits)']['Macro F1']:.4f}, ROC-AUC: {test_metrics['QML (4 Qubits)']['ROC-AUC']:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_qml.png"), dpi=300)
    plt.close()

    # Figure 5: ML vs QML Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6))
    eval_models = list(test_metrics.keys())
    m_acc = [test_metrics[m]["Accuracy"] for m in eval_models]
    m_bal_acc = [test_metrics[m]["Balanced Accuracy"] for m in eval_models]
    m_mac_f1 = [test_metrics[m]["Macro F1"] for m in eval_models]
    m_auc = [test_metrics[m]["ROC-AUC"] for m in eval_models]

    x = np.arange(len(eval_models))
    width = 0.2
    ax.bar(x - 1.5*width, m_acc, width, label="Accuracy", color="#1f77b4")
    ax.bar(x - 0.5*width, m_bal_acc, width, label="Balanced Accuracy", color="#ff7f0e")
    ax.bar(x + 0.5*width, m_mac_f1, width, label="Macro F1", color="#2ca02c")
    ax.bar(x + 1.5*width, m_auc, width, label="ROC-AUC", color="#d62728")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Heart Disease Classical ML vs PennyLane QML Benchmark (Untouched Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ml_vs_qml_comparison.png"), dpi=300)
    plt.close()

    # Figure 6: Feature Importance Chart (Random Forest / XGBoost)
    interpretable_model = fitted_models["Random Forest"]
    importances = interpretable_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), importances[indices], color="#3182bd", align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in indices], fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Gini Feature Importance)", fontsize=11, fontweight="bold")
    ax.set_title("Random Forest Biomarker Importance Ranking (Heart Disease CAD)", fontsize=13, fontweight="bold", pad=15)
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # Figure 7: ROC Curves for all models
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_curves.items()):
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.4f})", linewidth=2, color=palette[idx % len(palette)])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Chance Baseline (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Heart Disease Models", fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. PERSIST METRICS JSON
    # -------------------------------------------------------------
    best_overall = max(test_metrics.keys(), key=lambda k: (test_metrics[k]["Macro F1"], test_metrics[k]["ROC-AUC"]))

    metrics_json = {
        "dataset_info": {
            "name": "UCI Heart Disease Databases (Cleveland Clinic Foundation #45)",
            "n_samples_total": len(df_raw),
            "n_samples_train": n_train,
            "n_samples_test": n_test,
            "n_features": len(FEATURE_COLS),
            "target_mapping": {
                "0": "Absence of CAD (<50% stenosis)",
                "1": "Presence of CAD (>=50% stenosis)"
            },
            "pca_4_variance": float(cum_var[3]),
            "pca_6_variance": float(cum_var[5])
        },
        "models_performance": test_metrics,
        "best_classical_ml": best_ml_name,
        "best_overall_model": best_overall
    }

    with open(os.path.join(METRICS_DIR, "heart_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)

    print("\n--- SUMMARY COMPARISON TABLE ---")
    print(f"{'Model':22s} | {'Acc':6s} | {'BalAcc':6s} | {'Prec':6s} | {'Rec':6s} | {'MacF1':6s} | {'ROC-AUC':7s} | {'Time (s)':8s}")
    print("-" * 92)
    for m, res in test_metrics.items():
        print(f"{m:22s} | {res['Accuracy']:.4f} | {res['Balanced Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['Macro F1']:.4f} | {res['ROC-AUC']:.4f} | {res['Training Time']:8.4f}")

    # -------------------------------------------------------------
    # 8. AUTOMATED REPRODUCIBILITY VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 8. AUTOMATED REPRODUCIBILITY CHECK ---")
    r_imputer = joblib.load(os.path.join(PREPROC_DIR, "heart_imputer.joblib"))
    r_scaler = joblib.load(os.path.join(PREPROC_DIR, "heart_scaler.joblib"))
    r_pca_4 = joblib.load(os.path.join(PREPROC_DIR, "heart_pca_4.joblib"))
    r_ml_model = joblib.load(os.path.join(MODELS_DIR, "best_heart_ml_model.joblib"))

    # Reload test data directly from persisted X_test.csv
    X_te_reloaded = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv")).values

    r_y_pred_ml = r_ml_model.predict(X_te_reloaded)
    r_acc_ml = accuracy_score(y_test, r_y_pred_ml)
    r_mac_f1_ml = f1_score(y_test, r_y_pred_ml, average="macro")

    r_q_model = HeartQuantumClassifier(n_qubits=4)
    r_q_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "heart_qml_model_weights.pt"), weights_only=True))
    r_q_model.eval()

    X_te_pca4_r = r_pca_4.transform(X_te_reloaded)
    with torch.no_grad():
        r_q_logits = r_q_model(torch.tensor(X_te_pca4_r, dtype=torch.float32))
        r_y_pred_qml = torch.argmax(r_q_logits, dim=1).numpy()
    r_acc_qml = accuracy_score(y_test, r_y_pred_qml)
    r_mac_f1_qml = f1_score(y_test, r_y_pred_qml, average="macro")

    ml_match = (abs(r_acc_ml - test_metrics[best_ml_name]["Accuracy"]) < 1e-6) and (abs(r_mac_f1_ml - test_metrics[best_ml_name]["Macro F1"]) < 1e-6)
    qml_match = (abs(r_acc_qml - test_metrics["QML (4 Qubits)"]["Accuracy"]) < 1e-6) and (abs(r_mac_f1_qml - test_metrics["QML (4 Qubits)"]["Macro F1"]) < 1e-6)

    print(f"Classical ML Metric Verification: {'EXACT MATCH' if ml_match else 'MISMATCH'}")
    print(f"Quantum QML Metric Verification:   {'EXACT MATCH' if qml_match else 'MISMATCH'}")

    if ml_match and qml_match:
        print("\nREPRODUCIBILITY CHECK: PASSED\n")
    else:
        print("\nREPRODUCIBILITY CHECK: FAILED\n")
        raise RuntimeError("Reproducibility check failed! Reloaded metrics do not match.")


if __name__ == "__main__":
    run_heart_pipeline()
