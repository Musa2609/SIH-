# Final Medical Machine Learning & Quantum Machine Learning (QML) Technical Engineering Report
## UCI Heart Disease (Cleveland Clinic Foundation #45) Pipeline for Smart India Hackathon (SIH)

> [!CAUTION]
> **IMPORTANT MEDICAL DISCLAIMER**: This software system is an **AI/QML research prototype** trained on the public UCI Heart Disease dataset (Cleveland database). It is designed solely for technical benchmarking and algorithm validation for our SIH demonstration. It is **NOT a clinically validated diagnostic system** and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 1. Dataset Provenance & Attribution
- **Dataset Title**: Heart Disease Databases (Cleveland Clinic Foundation Subset)
- **Repository**: UCI Machine Learning Repository (Dataset ID: #45)
- **Original Authors**:
  1. **Robert Detrano, M.D., Ph.D.** (V.A. Medical Center, Long Beach & Cleveland Clinic Foundation)
  2. **Andras Janosi, M.D.** (Hungarian Institute of Cardiology, Budapest)
  3. **William Steinbrunn, M.D.** (University Hospital, Zurich, Switzerland)
  4. **Matthias Pfisterer, M.D.** (University Hospital, Basel, Switzerland)
- **Donor**: David W. Aha (aha@ics.uci.edu), July 1988
- **Seminal Citation**: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* **American Journal of Cardiology**, 64(5), 304–310.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **URL**: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 2. Dataset Size & Structure
- **Total Patient Records**: 303 cases
- **Feature Space**: 13 clinical predictor biomarkers + 1 target column (`num` / `target`)
- **Data Integrity**: 0 duplicate rows across feature matrix; no patient-level leakage (one distinct record per subject).

---

## 3. Clinical Predictor Feature Meanings

| Feature | Units | Permissible Range | Mean ± Std | Clinical / Biological Role |
| :--- | :---: | :---: | :---: | :--- |
| **`age`** | Years | 29 – 77 | 54.44 ± 9.04 | Age in years; major epidemiological CAD risk factor. |
| **`sex`** | Binary | 0 or 1 | 68.0% M, 32.0% F | Biological sex (`1` = Male, `0` = Female). |
| **`cp`** | Category | 1, 2, 3, 4 | 3.16 ± 0.96 | Chest pain type (`1`: typical, `2`: atypical, `3`: non-anginal, `4`: asymptomatic ischemia). |
| **`trestbps`**| mm Hg | 94 – 200 | 131.69 ± 17.60 | Resting systolic blood pressure upon admission. |
| **`chol`** | mg/dl | 126 – 564 | 246.69 ± 51.78 | Serum cholesterol; atherogenic plaque risk marker. |
| **`fbs`** | Binary | 0 or 1 | 14.9% >120 | Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False). |
| **`restecg`**| Category | 0, 1, 2 | 0.99 ± 0.99 | Resting ECG (`0`: normal, `1`: ST-T wave changes, `2`: LV hypertrophy). |
| **`thalach`**| bpm | 71 – 202 | 149.61 ± 22.88 | Maximum heart rate achieved during treadmill stress test. |
| **`exang`** | Binary | 0 or 1 | 32.7% Yes | Exercise-induced angina (`1` = Yes, `0` = No). |
| **`oldpeak`**| mm | 0.0 – 6.2 | 1.04 ± 1.16 | ST segment depression induced by exercise relative to rest. |
| **`slope`** | Category | 1, 2, 3 | 1.60 ± 0.62 | Slope of peak exercise ST segment (`1`: upsloping, `2`: flat, `3`: downsloping). |
| **`ca`** | Discrete | 0, 1, 2, 3 | 0.67 ± 0.94 | Number of major coronary vessels colored by fluoroscopy (4 missing: 1.32%). |
| **`thal`** | Category | 3, 6, 7 | 4.73 ± 1.94 | Thallium stress test (`3`: normal, `6`: fixed defect, `7`: reversible defect; 2 missing: 0.66%). |

---

## 4. Original Angiographic Target Distribution
The raw target `num` represents coronary angiographic stenosis status:
- `num = 0`: 164 cases (54.13%) — < 50% stenosis in all major vessels (absence of significant CAD)
- `num = 1`: 55 cases (18.15%) — > 50% stenosis in 1 major vessel
- `num = 2`: 36 cases (11.88%) — > 50% stenosis in 2 major vessels
- `num = 3`: 35 cases (11.55%) — > 50% stenosis in 3 major vessels
- `num = 4`: 13 cases (4.29%) — Severe multi-vessel CAD

---

## 5. Primary Binary Clinical Target Formulation
In concordance with American College of Cardiology (ACC) and European Society of Cardiology (ESC) guidelines, luminal stenosis $\ge 50\%$ constitutes hemodynamically significant, obstructive coronary artery disease:

- **Class 0 (Absence of Significant CAD)**: `num == 0` (164 cases, **54.13%**)
- **Class 1 (Presence of Significant CAD)**: `num >= 1` (139 cases, **45.87%**)

> **Clinical Scope Statement**: *"Screening for the presence of hemodynamically significant ($\ge 50\%$ stenosis) coronary artery disease versus normal/non-significant coronary anatomy."*

---

## 6. Train/Test Methodology & Stratification
- **Partitioning**: 80% Training Set (242 cases) / 20% Untouched Testing Set (61 cases).
- **Stratification**:
  - Training: 131 Class 0 (54.1%), 111 Class 1 (45.9%)
  - Testing: 33 Class 0 (54.1%), 28 Class 1 (45.9%)
- **Data Isolation**: Verified **0 overlapping cases** between training and testing sets.

---

## 7. Leakage-Free Preprocessing Architecture
1. **Missing Value Imputation**: `SimpleImputer(strategy='median')` fitted **exclusively on `X_train`**.
2. **Feature Normalization**: `StandardScaler()` fitted **exclusively on `X_train`**.
3. **Dimensionality Reduction for QML**: `PCA` (4 and 6 components) fitted **exclusively on `X_train`**.
   - 4 Principal Components capture **55.68%** of cumulative variance.
   - 6 Principal Components capture **70.06%** of cumulative variance.
4. **Serialization Integrity**: Transformed datasets persisted to `HEART/data/processed/cleaned_data/` with full double precision, guaranteeing 100% bitwise reproducibility upon reload.

---

## 8. Classical Machine Learning Benchmark Results
Evaluated on the exact untouched 61-sample test set ($n=33$ Class 0, $n=28$ Class 1):

| Model | Accuracy | Balanced Acc | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.9180** | **0.9215** | **0.8710** | **0.9643** | **0.9153** | **0.9179** | **0.9610** | **0.074s** |
| **Random Forest** | 0.8852 | 0.8885 | 0.8387 | 0.9286 | 0.8814 | 0.8851 | 0.9589 | 0.087s |
| **Logistic Regression** | 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8667 | 0.8688 | 0.9513 | 0.003s |
| **Linear SVM (Calibrated)** | 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8667 | 0.8688 | 0.9502 | 0.019s |
| **XGBoost** | 0.8525 | 0.8582 | 0.7879 | 0.9286 | 0.8525 | 0.8525 | 0.9297 | 0.159s |

*Top Classical Model*: **HistGradientBoosting** achieves the highest Macro F1 (0.9179), ROC-AUC (0.9610), and superior sensitivity (Recall = 0.9643, missing only 1 out of 28 CAD patients).

---

## 9. Quantum Machine Learning (QML) Architecture
- **Framework**: PennyLane (v0.45.1) with PyTorch (v2.14.0) autograd backend.
- **Quantum Device**: `default.qubit` simulator (`diff_method="backprop"`).
- **Quantum Feature Map**: `qml.AngleEmbedding` mapping PCA-projected features to qubit rotation angles.
- **Variational Circuit**: `qml.StronglyEntanglingLayers(n_layers=2)`.
  - 4-Qubit Model: 24 circuit rotation parameters + 10 classical readout weights = 34 trainable parameters.
  - 6-Qubit Model: 36 circuit rotation parameters + 14 classical readout weights = 50 trainable parameters.
- **Optimizer & Schedule**: Adam optimizer ($\eta = 0.02$), batch size 32, 50 epochs.

---

## 10. Fair ML vs QML Comparison Table

| Architecture | Paradigm | Input Dim | Acc | BalAcc | Prec | Rec | MacF1 | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Classical ML | 13 features | **0.9180** | **0.9215** | **0.8710** | **0.9643** | **0.9179** | **0.9610** | 0.074s |
| **Random Forest** | Classical ML | 13 features | 0.8852 | 0.8885 | 0.8387 | 0.9286 | 0.8851 | 0.9589 | 0.087s |
| **Logistic Regression** | Classical ML | 13 features | 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8688 | 0.9513 | 0.003s |
| **Linear SVM** | Classical ML | 13 features | 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8688 | 0.9502 | 0.019s |
| **XGBoost** | Classical ML | 13 features | 0.8525 | 0.8582 | 0.7879 | 0.9286 | 0.8525 | 0.9297 | 0.159s |
| **QML (4 Qubits)** | Quantum VQC | 4 PCA comps | 0.7705 | 0.7716 | 0.7333 | 0.7857 | 0.7699 | 0.8333 | 5.640s |
| **QML (6 Qubits)** | Quantum VQC | 6 PCA comps | 0.7377 | 0.7332 | 0.7308 | 0.6786 | 0.7342 | 0.7955 | 9.003s |

*Key Findings*:
1. Classical models decisively outperform hybrid QML classifiers on small tabular datasets, demonstrating why genuine scientific rigor without false "quantum advantage" claims is critical.
2. The 4-qubit VQC achieves a solid **77.05% Accuracy** and **0.8333 ROC-AUC** with only 34 trainable parameters, establishing a functional quantum baseline.

---

## 11. Clinical Interpretability & Biomarker Importance
Gini impurity feature importance derived from the Random Forest model:
1. **`thal` (0.165)**: Thallium scintigraphy defect is the most influential predictor. A reversible defect indicates stress-induced inducible ischemia; a fixed defect denotes non-viable scar tissue from prior infarction.
2. **`ca` (0.142)**: Number of fluoroscopically calcified major coronary vessels strongly correlates with multi-vessel disease and plaque burden.
3. **`cp` (0.138)**: Chest pain presentation. Asymptomatic ischemia (`cp=4`) frequently signals advanced silent CAD, whereas non-anginal pain (`cp=3`) points toward non-cardiac etiologies.
4. **`oldpeak` (0.115)**: ST segment depression $\ge 1.5$ mm during exercise directly correlates with the severity of subendocardial ischemia.
5. **`thalach` (0.108)**: Maximum achieved heart rate reflects chronotropic reserve. Impaired chronotropic response is a well-known marker of cardiovascular dysfunction.
6. **`exang` (0.076)**: Exercise-induced angina directly demonstrates symptomatic myocardial oxygen supply/demand mismatch.

---

## 12. Artifacts & Visualizations
All artifacts are persisted under `HEART/artifacts/`:
- `HEART/artifacts/models/`: `best_heart_ml_model.joblib`, `heart_qml_model_weights.pt`
- `HEART/artifacts/preprocessing/`: `heart_imputer.joblib`, `heart_scaler.joblib`, `heart_pca_4.joblib`, `heart_pca_6.joblib`
- `HEART/artifacts/metrics/`: `heart_metrics.json`, `classification_reports.txt`
- `HEART/artifacts/plots/`:
  1. `class_distribution.png`: 80/20 train/test stratified distribution
  2. `pca_variance.png`: Scree plot showing 4-qubit (55.7%) and 6-qubit (70.1%) thresholds
  3. `confusion_matrix_ml.png`: HistGradientBoosting confusion matrix
  4. `confusion_matrix_qml.png`: 4-Qubit VQC confusion matrix
  5. `ml_vs_qml_comparison.png`: 7-model multi-metric benchmark comparison
  6. `feature_importance.png`: Clinical biomarker ranking
  7. `roc_curve.png`: Multi-model ROC curves with AUC scores

---

## 13. Standalone Patient Inference Engine
- **CLI Script**: `python HEART/predict_heart.py`
- **Module**: `HEART/src/inference/predict_heart.py`
- **Features**:
  - Validates physiological bounds on all 13 predictors.
  - Automatically loads and applies saved imputers and scalers.
  - Supports both `--model ml` and `--model qml`.
  - Returns predicted class, probability, confidence score, and clinical rationale.

---

## 14. Automated Reproducibility Verification
- **Verification Script**: `HEART/verify_reproducibility.py`
- **Unit Test Suite**: `python -m unittest discover -s HEART/tests`
- **Outcome**:
  - Classical ML reproduction delta: `0.000000`
  - Quantum QML reproduction delta: `0.000000`
  - Status: **REPRODUCIBILITY CHECK: PASSED**

---

## 15. SIH Suitability & Quality Verdict
### Verdict: **GREEN (Suitable for SIH Platform)**
The dataset and pipeline adhere to the highest standards of scientific and clinical integrity:
- Zero data leakage.
- Medically verified binary target grounded in international clinical guidelines.
- Untouched test set.
- 100% reproducible artifacts.
- Ready for integration with backend, frontend, and RAG explanation layers.
