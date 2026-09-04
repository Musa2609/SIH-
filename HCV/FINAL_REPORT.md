# Final Medical Machine Learning & Quantum Machine Learning (QML) Technical Engineering Report
## UCI Hepatitis C Virus (HCV) Dataset (#571) Pipeline for Smart India Hackathon (SIH)

> [!CAUTION]
> **IMPORTANT MEDICAL DISCLAIMER**: This software system is an **AI/QML research prototype** trained on the public UCI Hepatitis C Virus dataset. It is designed solely for technical benchmarking and algorithm validation for our SIH demonstration. It is **NOT a clinically validated diagnostic system** and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 1. Dataset Provenance & Attribution
- **Dataset Title**: UCI Hepatitis C Virus (HCV) Dataset (#571)
- **Repository**: UCI Machine Learning Repository
- **Original Citation / Source**: Lichtinghagen et al. (2013), *"Serum levels of collagen IV and laminin in patients with chronic hepatitis C"*, Clinical Chemistry and Laboratory Medicine (CCLM).
- **Dataset ID**: #571 (`hcvdat0.csv`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Dataset Size & Structure
- **Total Patient Records**: 615 samples
- **Feature Space**: 12 clinical predictor biomarkers + 1 target category column (`Category`)
- **Data Integrity**: 0 duplicate rows across the feature matrix; ID column `Unnamed: 0` removed.

---

## 3. Clinical Predictor Feature Meanings

| Feature | Units | Data Type | Clinical Meaning & Biological Role |
| :--- | :---: | :---: | :--- |
| **`Age`** | Years | Float | Patient age in years |
| **`Sex`** | Binary | Categorical (`'m'`/`'f'`) | Biological sex (Encoded `'m'`: 1, `'f'`: 0) |
| **`ALB`** | g/L | Float | Albumin — Liver protein synthesis indicator |
| **`ALP`** | U/L | Float | Alkaline Phosphatase — Biliary tract obstruction marker |
| **`ALT`** | U/L | Float | Alanine Transaminase — Specific liver cell damage biomarker |
| **`AST`** | U/L | Float | Aspartate Transaminase — Hepatocellular inflammation/necrosis marker |
| **`BIL`** | umol/L | Float | Total Bilirubin — Hepatic clearance & jaundice marker |
| **`CHE`** | kU/L | Float | Cholinesterase — Functional liver parenchyma synthesis capacity |
| **`CHOL`** | mmol/L | Float | Total Cholesterol — Hepatic lipid metabolism |
| **`CREA`** | umol/L | Float | Serum Creatinine — Kidney function / Hepatorenal syndrome marker |
| **`GGT`** | U/L | Float | Gamma-Glutamyl Transferase — Biliary disease & alcohol/drug toxicity marker |
| **`PROT`** | g/L | Float | Total Protein — Serum protein homeostasis |

---

## 4. Original 5 UCI Categories & Sample Breakdown
The raw dataset contains 5 fine-grained clinical labels:
1. `0=Blood Donor`: 533 samples (86.67%)
2. `0s=suspect Blood Donor`: 7 samples (1.14%)
3. `1=Hepatitis`: 24 samples (3.90%)
4. `2=Fibrosis`: 21 samples (3.41%)
5. `3=Cirrhosis`: 30 samples (4.88%)

---

## 5. Primary Binary Clinical Target Mapping
To establish a robust binary screening task, the 5 categories were mapped into two clinically meaningful cohorts:

- **Class 0 (Healthy/Control)**:
  - Mapped from `0=Blood Donor` (533) + `0s=suspect Blood Donor` (7)
  - **Total Class 0**: **540 samples (87.80%)**
- **Class 1 (HCV-Related Pathology)**:
  - Mapped from `1=Hepatitis` (24) + `2=Fibrosis` (21) + `3=Cirrhosis` (30)
  - **Total Class 1**: **75 samples (12.20%)**

> **Clinical Scope Statement**: *"Classification of healthy/control samples versus HCV-related liver pathology."* (This binary model screens for HCV pathology vs controls; it does not claim to differentiate exact disease stages).

---

## 6. Train/Test Methodology & Stratification
- **Partition Ratio**: 80% Training Set (492 samples), 20% Untouched Testing Set (123 samples).
- **Stratification**: Exact ratio preserved across both splits:
  - **Train Set** (n=492): Class 0 = 432 (87.80%), Class 1 = 60 (12.20%)
  - **Test Set** (n=123): Class 0 = 108 (87.80%), Class 1 = 15 (12.20%)
- **Random Seed**: `42` fixed globally for exact reproducibility.

---

## 7. Strict Data Leakage Prevention Checks
- **Zero Cross-Set Overlap**: Cross-set inner join check confirms **0 overlapping samples** between training and test sets.
- **Transformer Isolation**: All preprocessing objects (`SimpleImputer`, `Sex` Binary Encoder, `StandardScaler`, `PCA`) were fitted **STRICTLY on `X_train`**. `X_test` was only transformed using fitted parameters.
- **Untouched Test Set**: The 123-sample test set remained strictly isolated until final inference evaluation.

---

## 8. Preprocessing Details
1. **Missing Value Handling**: Imputed 31 total missing values across `ALB` (1), `ALP` (18), `ALT` (1), `CHOL` (10), and `PROT` (1) using `SimpleImputer(strategy='median')` fitted strictly on training data.
2. **Categorical Encoding**: Binary mapping for `Sex` (`'m'`: 1, `'f'`: 0).
3. **Standardization**: `StandardScaler` fitted strictly on imputed training features.
4. **QML Feature Compression**: `PCA` fitted strictly on standardized training data. 4-component PCA captures **58.43% variance**; 6-component PCA captures **73.96% variance**.

---

## 9. Classical Machine Learning Model Performance

Evaluated on the untouched 123-sample test set:

| Model | Accuracy | Balanced Acc | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Top Model)** | **98.37%** | **0.9333** | **1.0000** | **0.8667** | **0.9286** | **0.9597** | **0.9963** | 0.1771s |
| **Random Forest** | 97.56% | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.9376 | **0.9972** | 0.1453s |
| **HistGradientBoosting** | 96.75% | 0.8954 | 0.9231 | 0.8000 | 0.8571 | 0.9194 | 0.9957 | 1.5883s |
| **Logistic Regression** | 95.12% | 0.8000 | 1.0000 | 0.6000 | 0.7500 | 0.8615 | 0.9235 | 0.0166s |
| **Linear SVM** | 94.31% | 0.7667 | 1.0000 | 0.5333 | 0.6957 | 0.8321 | 0.9778 | 0.0423s |

- **Top Classical Model**: **XGBoost** (Macro F1 = **0.9597**, ROC-AUC = **0.9963**, Precision = **100.0%**, Recall = **86.67%**).

---

## 10. Quantum Machine Learning (QML) Architecture & Performance

### Architecture
- **Simulator**: PennyLane `default.qubit` simulator integrated with PyTorch `nn.Module`.
- **Feature Map**: `qml.AngleEmbedding(features, wires=range(n_qubits))`
- **Ansatz**: `qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))` (2 layers)
- **Readout**: Linear layer (`nn.Linear(n_qubits, 2)`) mapping quantum expectation values to binary class logits.
- **Optimizer & Loss**: Adam (`lr=0.02`), `nn.CrossEntropyLoss()`, 50 epochs with `diff_method="backprop"`.

### Performance Results
| QML Model | Qubits | Variational Layers | Trainable Parameters | PCA Variance | Test Accuracy | Balanced Acc | Macro F1 | ROC-AUC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QML (4 Qubits)** | 4 | 2 | 34 | 58.43% | **91.87%** | **0.7528** | **0.7850** | **0.9290** | 24.56s |
| **QML (6 Qubits)** | 6 | 2 | 48 | 73.96% | 89.43% | 0.6528 | 0.6882 | 0.9333 | 39.10s |

---

## 11. Confusion Matrices Summary

### Best Classical ML Model (XGBoost)
- True Healthy (108): **108 Correct**, **0 False Positives** (100% Specificity!)
- True HCV Pathology (15): **13 Correct**, **2 False Negatives** (86.67% Sensitivity!)

### QML 4-Qubit Model
- True Healthy (108): **105 Correct**, **3 False Positives**
- True HCV Pathology (15): **8 Correct**, **7 False Negatives**

---

## 12. Clinical Feature Importance Ranking

Gini feature importances from the top tree ensemble model (Random Forest / XGBoost):

1. **`AST` (Aspartate Transaminase)**: **0.285** (Dominant hepatocellular damage biomarker)
2. **`ALT` (Alanine Transaminase)**: **0.182** (Primary liver enzyme marker)
3. **`GGT` (Gamma-Glutamyl Transferase)**: **0.147** (Biliary pathology indicator)
4. **`BIL` (Bilirubin)**: **0.104** (Hepatic excretion indicator)
5. **`CHE` (Cholinesterase)**: **0.088** (Functional parenchymal synthesis marker)
6. **`ALB` (Albumin)**: **0.061** (Protein synthesis capacity)
7. **`ALP` (Alkaline Phosphatase)**: **0.043**
8. **`CREA`, `CHOL`, `Age`, `PROT`, `Sex`**: Secondary metabolic parameters ($<0.030$ each).

> **Clinical Interpretation**: Liver enzyme markers `AST`, `ALT`, `GGT`, and `BIL` drive over 70% of model predictions. This closely aligns with established hepatology diagnostic guidelines.

---

## 13. PCA & Dimensionality Reduction Analysis
- Component 1: 20.78% variance (Driven by liver enzymes `AST`, `ALT`, `GGT`)
- Component 2: 15.83% variance (Driven by proteins `ALB`, `PROT`, `CHE`)
- Component 3: 12.11% variance (Driven by `BIL` and `CREA`)
- Component 4: 9.71% variance
- **PCA-4 (4 Qubits)**: 58.43% cumulative variance
- **PCA-6 (6 Qubits)**: 73.96% cumulative variance

---

## 14. Dataset & Pipeline Limitations
1. **Class Imbalance**: HCV Pathology represents 12.2% of patient records (Imbalance Ratio 7.2:1).
2. **Sample Size**: 615 total records. While sufficient for SIH technical prototyping, clinical deployment requires multi-center validation on larger cohorts.
3. **QML Feature Compression**: Reducing 12 features to 4 principal components discards 41.57% of information, creating a bottleneck for the quantum classifier relative to classical gradient boosting.

---

## 15. Reproducibility Test Results
Automated reloading test executed on saved artifacts in [`HCV/artifacts/`](file:///c:/Users/HP/Desktop/SIH/HCV/artifacts/):
- Preprocessing objects reloaded: `hcv_imputer.joblib`, `hcv_sex_encoder.joblib`, `hcv_scaler.joblib`, `hcv_pca_4.joblib`
- Models reloaded: `best_hcv_ml_model.joblib`, `hcv_qml_model_weights.pt`
- Re-evaluated test set Accuracy: **98.37% (ML)** / **91.87% (QML 4-qubit)**
- Re-evaluated test set Macro F1: **0.9597 (ML)** / **0.7850 (QML 4-qubit)**

```
REPRODUCIBILITY CHECK: PASSED
```

---

## 16. Final 10-Point Audit Checklist & Verdict

| Audit Question | Explicit Verdict | Evidence / Details |
| :--- | :---: | :--- |
| **1. Is the dataset genuine and medically documented?** | **YES** | Official UCI Dataset #571 (`hcvdat0.csv`) from Lichtinghagen et al. (2013). |
| **2. Is the binary target mapping correct?** | **YES** | Class 0: Healthy/Control (540); Class 1: HCV Pathology (75). |
| **3. Is there any data leakage?** | **NO** | Zero train/test sample overlap; all transformers fitted strictly on `X_train`. |
| **4. Was the test set completely untouched?** | **YES** | 123-sample test set reserved strictly for final inference. |
| **5. Is the ML result reproducible?** | **YES** | Reloading `best_hcv_ml_model.joblib` yields exact 98.37% accuracy. |
| **6. Is the QML result reproducible?** | **YES** | Reloading `hcv_qml_model_weights.pt` yields exact 91.87% accuracy. |
| **7. What is the best ML model?** | **XGBoost** | Accuracy: **98.37%**, Macro F1: **0.9597**, ROC-AUC: **0.9963**. |
| **8. What is the best QML model?** | **4-Qubit PennyLane VQC** | Accuracy: **91.87%**, Macro F1: **0.7850**, ROC-AUC: **0.9290**. |
| **9. Is the ML vs QML comparison fair?** | **YES** | Evaluated on the exact same 123-sample untouched test set with identical metrics. |
| **10. Is this suitable for our SIH prototype and presentation?** | **YES** | High technical quality, 100% reproducible, medically aligned feature importances. |
