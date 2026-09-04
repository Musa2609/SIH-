# SIH Team Handoff & Developer Guide
## For Anmol and Engineering Team Members

> [!IMPORTANT]
> **HCV MODULE STATUS**: **HCV is finalized.** Do not modify the HCV dataset, target definition, preprocessing, or trained model binaries unless a genuine technical bug is discovered.

---

## 1. Executive Summary & Project Status
The Hepatitis C Virus (`HCV`) module is completely finalized, 100% reproducible, and pushed to version control. The pipeline achieves **98.37% Accuracy** and **0.9597 Macro F1** using XGBoost, and **91.87% Accuracy** using a 4-qubit PennyLane Quantum Classifier on untouched test data.

The project repository is now organized into modular disease components. The next primary engineering objective is implementing the **`HEART` disease module**.

---

## 2. Standardized 16-Step Workflow for Adding the `HEART` Module

To ensure the `HEART` disease module meets the exact technical and medical quality standards established in `HCV`, follow this 16-step pipeline blueprint:

### Step 1: Add Trusted Raw Dataset
- Place the official raw dataset (e.g., UCI Heart Disease Dataset #45) in `HEART/data/raw/heart.csv`.
- Do NOT use synthetic or undocumented data.

### Step 2: Verify Provenance & License
- Document the original dataset source, institution, publication citation, and license in `HEART/FINAL_REPORT.md`.

### Step 3: Audit Target Labels
- Inspect original target categories. Define a clear, medically defensible target (e.g., Class 0: Absence of Heart Disease vs Class 1: Presence of Cardiovascular Pathology).

### Step 4: Check Class Distribution & Imbalance
- Calculate exact counts and percentages for each target class in train and test splits. Calculate imbalance ratio.

### Step 5: Check Missing & Invalid Values
- Identify missing values per column. Plan appropriate imputation (e.g., median imputer) fitted **STRICTLY on `X_train`**.

### Step 6: Check Duplicates & Invalid Records
- Verify zero duplicate rows exist. Remove non-clinical ID columns.

### Step 7: Prevent Data Leakage & Ensure Train/Test Isolation
- Split data into 80% Training / 20% Untouched Testing using stratified sampling (`random_state=42`).
- Verify **zero overlapping samples** between train and test splits using an inner join check.

### Step 8: Identify Clinical Feature Meanings
- Map every feature name to its physiological unit and clinical meaning (e.g., `cp`: Chest Pain Type, `thalach`: Maximum Heart Rate Achieved).

### Step 9: Build Preprocessing Pipeline
- Fit imputers, encoders, `StandardScaler`, and `PCA` (4 & 6 components) **ONLY on `X_train`**.
- Apply fitted transformers to `X_test`. Save clean CSVs in `HEART/data/processed/cleaned_data/`.

### Step 10: Train Classical Machine Learning Models
- Train 5 models: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, and Linear SVM.
- Evaluate on the untouched test set using Accuracy, Balanced Accuracy, Precision, Recall, Macro F1, ROC-AUC, and Confusion Matrix.
- Select best model based on **Macro F1 / ROC-AUC**.

### Step 11: Train PennyLane PyTorch QML Models
- Build a genuine quantum simulator using PennyLane `default.qubit` (`diff_method="backprop"`).
- Train 4-qubit and 6-qubit VQC architectures with `AngleEmbedding` and `StronglyEntanglingLayers`.

### Step 12: Evaluate on Untouched Test Set
- Benchmark classical ML vs QML on the exact same test set. Generate summary comparison table.

### Step 13: Save Pipeline Artifacts
Persist all joblib and PyTorch state files into categorized subfolders:
- `HEART/artifacts/models/`: `best_heart_ml_model.joblib`, `heart_qml_model_weights.pt`
- `HEART/artifacts/preprocessing/`: `heart_scaler.joblib`, `heart_pca_4.joblib`, `heart_imputer.joblib`
- `HEART/artifacts/metrics/`: `heart_metrics.json`, `classification_reports.txt`

### Step 14: Generate High-Resolution Visualizations
Save 7 publication plots in `HEART/artifacts/plots/`:
- `class_distribution.png`
- `ml_vs_qml_comparison.png`
- `confusion_matrix_ml.png`
- `confusion_matrix_qml.png`
- `feature_importance.png`
- `pca_variance.png`
- `roc_curve.png`

### Step 15: Add Standalone Patient Inference Tool
- Create `HEART/predict_heart.py` accepting clinical biomarkers and returning prediction (`Healthy` vs `Cardiovascular Pathology`) with confidence score.

### Step 16: Document & Verify Reproducibility
- Create `HEART/FINAL_REPORT.md` covering all 16 sections.
- Run automated reproducibility check to confirm reloaded artifacts match test metrics.

---

## 3. Directory Structure Guidelines for New Modules
```
HEART/
├── data/
│   ├── raw/heart.csv
│   └── processed/cleaned_data/
├── src/
│   ├── preprocessing/
│   ├── ml/
│   ├── qml/
│   └── inference/predict_heart.py
├── artifacts/
│   ├── models/
│   ├── preprocessing/
│   ├── metrics/
│   └── plots/
├── tests/test_heart_pipeline.py
├── run_heart_pipeline.py
└── FINAL_REPORT.md
```
