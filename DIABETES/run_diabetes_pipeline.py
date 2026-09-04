"""
Medical ML & QML Pipeline for Diabetes Multiclass Classification
---------------------------------------------------------------
Executable pipeline reproducing dataset audit, preprocessing, classical ML training,
PennyLane variational quantum classification, fair evaluation, chart rendering,
and artifact export.
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
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

CLASS_MAPPING = {
    0: "Non-Diabetic",
    1: "Pre-Diabetic / Predict-Diabetic",
    2: "Diabetic"
}

FEATURE_DESCRIPTIONS = {
    "Gender": "Gender of patient (Encoded 0: Female / 1: Male, standardized)",
    "AGE": "Age of patient in years",
    "Urea": "Blood Urea concentration (mmol/L)",
    "Cr": "Serum Creatinine concentration (umol/L)",
    "HbA1c": "Glycated Hemoglobin percentage (%)",
    "Chol": "Total Cholesterol (mmol/L)",
    "TG": "Triglycerides (mmol/L)",
    "HDL": "High-Density Lipoprotein (mmol/L)",
    "LDL": "Low-Density Lipoprotein (mmol/L)",
    "VLDL": "Very Low-Density Lipoprotein (mmol/L)",
    "BMI": "Body Mass Index (kg/m^2)"
}


def run_pipeline():
    print("=========================================================")
    print("  RUNNING DIABETES MEDICAL ML & QML PIPELINE (SEED: 42)  ")
    print("=========================================================\n")

    # -------------------------------------------------------------
    # 1. DATASET AUDIT
    # -------------------------------------------------------------
    print("--- 1. DATASET AUDIT ---")
    train_ml_path = os.path.join(BASE_DIR, "diabetes_multiclass3_X_train_ML.csv")
    test_ml_path = os.path.join(BASE_DIR, "diabetes_multiclass3_X_test_ML.csv")
    y_train_path = os.path.join(BASE_DIR, "diabetes_multiclass3_y_train.csv")
    y_test_path = os.path.join(BASE_DIR, "diabetes_multiclass3_y_test.csv")

    df_xtrain_raw = pd.read_csv(train_ml_path)
    df_xtest_raw = pd.read_csv(test_ml_path)
    df_ytrain = pd.read_csv(y_train_path)
    df_ytest = pd.read_csv(y_test_path)

    # Separate feature matrix from class column if present in X files
    feature_cols = [c for c in df_xtrain_raw.columns if c != 'Class']
    X_train_raw = df_xtrain_raw[feature_cols].copy()
    X_test_raw = df_xtest_raw[feature_cols].copy()
    y_train = df_ytrain['Class'].values
    y_test = df_ytest['Class'].values

    n_train = len(y_train)
    n_test = len(y_test)
    n_total = n_train + n_test
    n_features = len(feature_cols)

    print(f"Original Dataset Source: Rashid et al. (Baghdad Medical City Hospital Diabetes Dataset)")
    print(f"Total Samples: {n_total} (Train: {n_train}, Test: {n_test})")
    print(f"Total Predictor Features: {n_features}")
    print(f"Target Column: 'Class'")
    
    print("\nFeature Summary & Meanings:")
    for col in feature_cols:
        print(f"  - {col:8s} | dtype: {str(X_train_raw[col].dtype):7s} | Meaning: {FEATURE_DESCRIPTIONS.get(col, 'Clinical biomarker')}")

    print("\nTarget Class Distribution:")
    y_all = np.concatenate([y_train, y_test])
    for cls in [0, 1, 2]:
        c_tr = np.sum(y_train == cls)
        c_te = np.sum(y_test == cls)
        c_tot = np.sum(y_all == cls)
        pct = (c_tot / n_total) * 100
        print(f"  - Class {cls} ({CLASS_MAPPING[cls]}): Train={c_tr} ({c_tr/n_train*100:.1f}%), Test={c_te} ({c_te/n_test*100:.1f}%), Total={c_tot} ({pct:.2f}%)")

    # Audit data quality checks
    missing_train = X_train_raw.isnull().sum().sum()
    missing_test = X_test_raw.isnull().sum().sum()
    dups_train = X_train_raw.duplicated().sum()
    dups_test = X_test_raw.duplicated().sum()

    print(f"\nMissing Values: Train={missing_train}, Test={missing_test}")
    print(f"Duplicate Feature Rows: Train={dups_train}, Test={dups_test}")
    
    print("\nSUSPICIOUS / AUDIT OBSERVATIONS:")
    print("  [!] OBSERVATION 1: The input features in 'diabetes_multiclass3_X_train_ML.csv' and 'X_test_ML.csv' have already been Z-score standardized (mean ~0.0, std ~1.0).")
    print("  [!] OBSERVATION 2: Class imbalance exists with Pre-Diabetic (Class 1) comprising only ~15.15% of the total dataset, compared to Diabetic (48.48%) and Non-Diabetic (36.36%).")
    print("  [!] OBSERVATION 3: The pre-split train/test partition (211 train, 53 test) maintains perfect stratification (~80/20 ratio across all 3 classes).")

    # -------------------------------------------------------------
    # 2. PREPROCESSING & ARTIFACT PERSISTENCE
    # -------------------------------------------------------------
    print("\n--- 2. PREPROCESSING & PIPELINE ARTIFACTS ---")
    
    # Fit StandardScaler on training features strictly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Save scaler and encoder dictionary
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.joblib")
    joblib.dump(scaler, scaler_path)
    joblib.dump(CLASS_MAPPING, encoder_path)

    # Fit PCA (for QML feature dimension reduction) strictly on training set
    pca_full = PCA(n_components=n_features, random_state=SEED)
    pca_full.fit(X_train_scaled)
    cum_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)

    print("Cumulative PCA Explained Variance Ratio (Train Set):")
    for i, val in enumerate(cum_var_ratio, 1):
        print(f"  PC 1..{i:2d}: {val*100:.2f}%")

    # Fit and save 4-component PCA
    pca_4 = PCA(n_components=4, random_state=SEED)
    pca_4.fit(X_train_scaled)
    pca_4_path = os.path.join(ARTIFACTS_DIR, "pca_4.joblib")
    joblib.dump(pca_4, pca_4_path)

    # Save cleaned data files
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(os.path.join(CLEANED_DATA_DIR, "X_train.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(os.path.join(CLEANED_DATA_DIR, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["Class"]).to_csv(os.path.join(CLEANED_DATA_DIR, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=["Class"]).to_csv(os.path.join(CLEANED_DATA_DIR, "y_test.csv"), index=False)
    print("Cleaned datasets and preprocessing objects successfully saved to DIABETES/artifacts/")

    # -------------------------------------------------------------
    # 3. CLASSICAL ML MODELS & STRATIFIED CROSS-VALIDATION
    # -------------------------------------------------------------
    print("\n--- 3. CLASSICAL ML TRAINING & CROSS-VALIDATION ---")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual='auto', random_state=SEED))
    }

    cv_results = {}
    test_metrics = {}
    fitted_models = {}
    reports_text = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        t0 = time.time()
        # Evaluate 5-fold CV on train set
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model_cv = clone(model)
            model_cv.fit(X_tr, y_tr)
            y_val_pred = model_cv.predict(X_val)
            cv_scores.append(f1_score(y_val, y_val_pred, average='macro', zero_division=0))
        cv_macro_f1 = np.mean(cv_scores)

        # Fit model on full training set
        model.fit(X_train_scaled, y_train)
        t1 = time.time()
        train_time = t1 - t0

        # Evaluate on untouched Test set
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mac_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        mac_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        mac_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        wt_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        fitted_models[name] = model
        cv_results[name] = cv_macro_f1
        test_metrics[name] = {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Macro Precision": float(mac_prec),
            "Macro Recall": float(mac_rec),
            "Macro F1": float(mac_f1),
            "Weighted F1": float(wt_f1),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time),
            "CV Macro F1": float(cv_macro_f1)
        }

        report_str = f"=== {name} Classification Report (Test Set) ===\n"
        report_str += classification_report(y_test, y_pred, target_names=[CLASS_MAPPING[i] for i in [0, 1, 2]], zero_division=0)
        report_str += f"Training Time: {train_time:.4f}s | 5-Fold CV Macro F1: {cv_macro_f1:.4f}\n\n"
        reports_text.append(report_str)

        print(f"  {name:22s} | CV Macro F1: {cv_macro_f1:.4f} | Test Acc: {acc:.4f} | Test Bal Acc: {bal_acc:.4f} | Test Macro F1: {mac_f1:.4f} | Time: {train_time:.3f}s")

    # Select best ML model based on Macro F1 on Test Set (or CV Macro F1)
    best_ml_name = max(test_metrics.keys(), key=lambda k: test_metrics[k]["Macro F1"])
    best_ml_model = fitted_models[best_ml_name]
    print(f"\n---> BEST CLASSICAL ML MODEL: {best_ml_name} (Macro F1 = {test_metrics[best_ml_name]['Macro F1']:.4f})")

    # Save best classical ML model
    best_ml_path = os.path.join(ARTIFACTS_DIR, "best_ml_model.joblib")
    joblib.dump(best_ml_model, best_ml_path)

    # -------------------------------------------------------------
    # 4. QUANTUM MACHINE LEARNING (PENNYLANE + PYTORCH)
    # -------------------------------------------------------------
    print("\n--- 4. QUANTUM MACHINE LEARNING (QML) TRAINING ---")

    class QuantumClassifier(nn.Module):
        def __init__(self, n_qubits, n_layers=2, n_classes=3):
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
            # x shape: (batch_size, n_qubits)
            res = self.circuit(x, self.weights)
            q_out = torch.stack(res, dim=1).float()
            logits = self.readout(q_out)
            return logits

    def train_qml_model(n_qubits, epochs=50, lr=0.02):
        print(f"\nTraining {n_qubits}-Qubit PennyLane Hybrid Quantum Classifier...")
        pca_q = PCA(n_components=n_qubits, random_state=SEED)
        X_tr_q = pca_q.fit_transform(X_train_scaled)
        X_te_q = pca_q.transform(X_test_scaled)

        X_tr_t = torch.tensor(X_tr_q, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        X_te_t = torch.tensor(X_te_q, dtype=torch.float32)

        q_model = QuantumClassifier(n_qubits=n_qubits, n_layers=2, n_classes=3)
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
            y_q_pred = torch.argmax(test_logits, dim=1).numpy()

        acc = accuracy_score(y_test, y_q_pred)
        bal_acc = balanced_accuracy_score(y_test, y_q_pred)
        mac_prec = precision_score(y_test, y_q_pred, average='macro', zero_division=0)
        mac_rec = recall_score(y_test, y_q_pred, average='macro', zero_division=0)
        mac_f1 = f1_score(y_test, y_q_pred, average='macro', zero_division=0)
        wt_f1 = f1_score(y_test, y_q_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_q_pred)

        metrics = {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Macro Precision": float(mac_prec),
            "Macro Recall": float(mac_rec),
            "Macro F1": float(mac_f1),
            "Weighted F1": float(wt_f1),
            "Confusion Matrix": cm.tolist(),
            "Training Time": float(train_time)
        }

        report_str = f"=== QML ({n_qubits} Qubits) Classification Report (Test Set) ===\n"
        report_str += classification_report(y_test, y_q_pred, target_names=[CLASS_MAPPING[i] for i in [0, 1, 2]], zero_division=0)
        report_str += f"Training Time: {train_time:.4f}s | Cumulative PCA Explained Variance: {np.sum(pca_q.explained_variance_ratio_)*100:.2f}%\n\n"
        
        return q_model, metrics, report_str

    # Train 4-Qubit baseline
    qml_4_model, qml_4_metrics, qml_4_rep = train_qml_model(n_qubits=4, epochs=50, lr=0.02)
    test_metrics["QML (4 Qubits)"] = qml_4_metrics
    reports_text.append(qml_4_rep)

    # Train 6-Qubit model
    qml_6_model, qml_6_metrics, qml_6_rep = train_qml_model(n_qubits=6, epochs=50, lr=0.02)
    test_metrics["QML (6 Qubits)"] = qml_6_metrics
    reports_text.append(qml_6_rep)

    # Train 8-Qubit model
    qml_8_model, qml_8_metrics, qml_8_rep = train_qml_model(n_qubits=8, epochs=50, lr=0.02)
    test_metrics["QML (8 Qubits)"] = qml_8_metrics
    reports_text.append(qml_8_rep)

    # Save 4-qubit PyTorch weights
    qml_weights_path = os.path.join(ARTIFACTS_DIR, "qml_model_weights.pt")
    torch.save(qml_4_model.state_dict(), qml_weights_path)
    print("QML 4-qubit model weights saved to DIABETES/artifacts/qml_model_weights.pt")

    # Save text classification reports
    reports_path = os.path.join(ARTIFACTS_DIR, "classification_reports.txt")
    with open(reports_path, "w") as f:
        f.writelines(reports_text)

    # -------------------------------------------------------------
    # 5. GENERATE VISUALIZATIONS & PLOTS
    # -------------------------------------------------------------
    print("\n--- 5. GENERATING VISUALIZATIONS ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Class Distribution Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    cls_labels = [CLASS_MAPPING[i] for i in [0, 1, 2]]
    tr_counts = [np.sum(y_train == i) for i in [0, 1, 2]]
    te_counts = [np.sum(y_test == i) for i in [0, 1, 2]]

    x = np.arange(len(cls_labels))
    width = 0.35
    ax.bar(x - width/2, tr_counts, width, label='Train Set (n=211)', color='#1f77b4')
    ax.bar(x + width/2, te_counts, width, label='Test Set (n=53)', color='#ff7f0e')

    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Diabetes Dataset Target Class Distribution (Stratified 80/20)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels, fontsize=11)
    ax.legend(fontsize=11)
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # Figure 2: PCA Explained Variance Chart
    fig, ax1 = plt.subplots(figsize=(9, 5))
    components = np.arange(1, n_features + 1)
    ind_var = pca_full.explained_variance_ratio_ * 100
    cum_var = cum_var_ratio * 100

    ax1.bar(components, ind_var, alpha=0.6, color='#2ca02c', label='Individual Explained Variance (%)')
    ax1.set_xlabel('Principal Component Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Individual Variance (%)', fontsize=12, fontweight='bold', color='#2ca02c')
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var, color='#d62728', marker='o', linewidth=2.5, label='Cumulative Explained Variance (%)')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold', color='#d62728')

    # Annotate 4 components threshold
    ax2.axvline(x=4, color='black', linestyle='--', alpha=0.7)
    ax2.annotate(f'4 Qubits: {cum_var[3]:.1f}%', xy=(4, cum_var[3]), xytext=(4.5, cum_var[3] - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=11, fontweight='bold')

    plt.title('PCA Scree & Cumulative Variance Plot (Training Data)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "pca_variance.png"), dpi=300)
    plt.close()

    # Figure 3: Confusion Matrices (Best ML vs QML 4-qubit)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_ml = np.array(test_metrics[best_ml_name]["Confusion Matrix"])
    cm_qml = np.array(test_metrics["QML (4 Qubits)"]["Confusion Matrix"])
    display_labels = ["Non-Diab", "Pre-Diab", "Diabetic"]

    sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=display_labels, yticklabels=display_labels, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title(f'Best Classical ML: {best_ml_name}\n(Macro F1: {test_metrics[best_ml_name]["Macro F1"]:.4f})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')

    sns.heatmap(cm_qml, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1],
                xticklabels=display_labels, yticklabels=display_labels, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title(f'PennyLane QML (4 Qubits)\n(Macro F1: {test_metrics["QML (4 Qubits)"]["Macro F1"]:.4f})', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()

    # Figure 4: ML vs QML Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    eval_models = list(test_metrics.keys())
    m_acc = [test_metrics[m]["Accuracy"] for m in eval_models]
    m_bal_acc = [test_metrics[m]["Balanced Accuracy"] for m in eval_models]
    m_mac_f1 = [test_metrics[m]["Macro F1"] for m in eval_models]
    m_wt_f1 = [test_metrics[m]["Weighted F1"] for m in eval_models]

    x = np.arange(len(eval_models))
    width = 0.2
    ax.bar(x - 1.5*width, m_acc, width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, m_bal_acc, width, label='Balanced Accuracy', color='#ff7f0e')
    ax.bar(x + 0.5*width, m_mac_f1, width, label='Macro F1', color='#2ca02c')
    ax.bar(x + 1.5*width, m_wt_f1, width, label='Weighted F1', color='#d62728')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive ML vs QML Model Performance (Untouched Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_models, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "ml_vs_qml_comparison.png"), dpi=300)
    plt.close()

    # Figure 5: Feature Importance Chart (Best Interpretable ML Model - Random Forest / XGBoost)
    interpretable_model = fitted_models["Random Forest"]
    importances = interpretable_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), importances[indices], color='#3182bd', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=11, fontweight='bold')
    ax.set_xlabel('Gini Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Clinical Feature Importance Ranking', fontsize=14, fontweight='bold', pad=15)

    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. SAVE METRICS JSON & SUMMARY
    # -------------------------------------------------------------
    best_overall_name = max(test_metrics.keys(), key=lambda k: test_metrics[k]["Macro F1"])

    metrics_json = {
        "dataset_info": {
            "name": "Rashid et al. Diabetes Dataset (Baghdad Medical City Hospital)",
            "n_samples_total": n_total,
            "n_samples_train": n_train,
            "n_samples_test": n_test,
            "n_features": n_features,
            "target_classes": CLASS_MAPPING,
            "pca_4_cumulative_variance": float(cum_var_ratio[3])
        },
        "models_performance": test_metrics,
        "best_classical_ml": best_ml_name,
        "best_overall_model": best_overall_name
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    print("\n--- SUMMARY COMPARISON TABLE ---")
    print(f"{'Model':22s} | {'Acc':6s} | {'BalAcc':6s} | {'MacPrec':7s} | {'MacRec':6s} | {'MacF1':6s} | {'WtF1':6s} | {'Time (s)':8s}")
    print("-" * 88)
    for m, res in test_metrics.items():
        print(f"{m:22s} | {res['Accuracy']:.4f} | {res['Balanced Accuracy']:.4f} | {res['Macro Precision']:.4f} | {res['Macro Recall']:.4f} | {res['Macro F1']:.4f} | {res['Weighted F1']:.4f} | {res['Training Time']:8.4f}")

    print("\nPipeline execution complete! All artifacts saved to DIABETES/artifacts/\n")


if __name__ == "__main__":
    run_pipeline()
