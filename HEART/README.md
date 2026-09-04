# Heart Disease (Coronary Artery Disease) ML & QML Module

A fully audited, reproducible Classical Machine Learning and PennyLane Quantum Machine Learning (QML) pipeline for screening hemodynamically significant Coronary Artery Disease (CAD) using clinical biomarkers.

---

## 1. Dataset Source & Provenance
- **Dataset**: UCI Heart Disease Databases (Cleveland Clinic Foundation Subset)
- **Repository ID**: UCI Machine Learning Repository Dataset #45
- **Official URL**: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Primary Authors**: Robert Detrano, M.D., Ph.D., Andras Janosi, M.D., William Steinbrunn, M.D., Matthias Pfisterer, M.D.
- **Citation**: Detrano et al. (1989), *American Journal of Cardiology*, 64(5), 304–310.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Data Type**: 100% Real Clinical Catheterization Data (303 patient cases, 0 duplicates).

---

## 2. Feature Descriptions

The model utilizes 13 physiological and cardiological predictors:

| Feature | Units | Normal / Permissible Range | Clinical Role |
| :--- | :---: | :---: | :--- |
| `age` | Years | 29 – 77 | Age in years |
| `sex` | Binary | 0 or 1 | Biological sex (`1` = Male; `0` = Female) |
| `cp` | Category | 1, 2, 3, 4 | Chest pain type (`1`: typical, `2`: atypical, `3`: non-anginal, `4`: asymptomatic) |
| `trestbps` | mm Hg | 94 – 200 | Resting blood pressure on hospital admission |
| `chol` | mg/dl | 126 – 564 | Serum cholesterol |
| `fbs` | Binary | 0 or 1 | Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False) |
| `restecg` | Category | 0, 1, 2 | Resting ECG (`0`: normal, `1`: ST-T wave changes, `2`: LVH) |
| `thalach` | bpm | 71 – 202 | Maximum heart rate achieved on exercise stress test |
| `exang` | Binary | 0 or 1 | Exercise-induced angina (`1` = Yes, `0` = No) |
| `oldpeak` | mm | 0.0 – 6.2 | Exercise-induced ST depression relative to rest |
| `slope` | Category | 1, 2, 3 | Peak exercise ST slope (`1`: upsloping, `2`: flat, `3`: downsloping) |
| `ca` | Discrete | 0, 1, 2, 3 | Major coronary vessels colored by fluoroscopy |
| `thal` | Category | 3, 6, 7 | Thallium scintigraphy (`3`: normal, `6`: fixed scar, `7`: reversible defect) |

---

## 3. Medical Target Definition & Classification Task
- **Raw Variable**: `num` (Angiographic coronary stenosis degree, values 0 to 4).
- **Clinical Formulation**: In accordance with ACC/AHA clinical guidelines, $\ge 50\%$ diameter stenosis defines obstructive CAD:
  - **Class 0 (Absence of Significant CAD)**: `num == 0` (<50% diameter stenosis)
  - **Class 1 (Presence of Significant CAD)**: `num >= 1` ($\ge 50\%$ stenosis in $\ge 1$ major vessel)
- **Cohort Distribution**:
  - Total: 303 cases (Class 0: 164 [54.1%], Class 1: 139 [45.9%])
  - Imbalance Ratio: 1.18 : 1 (Naturally well-balanced)

---

## 4. Leakage-Free Preprocessing
- **Train/Test Split**: 80% Train (242 cases) / 20% Untouched Test (61 cases), stratified with `random_state=42`.
- **Imputation**: Median imputer fitted **strictly on `X_train`** (handles missing `ca` [4 cases] and `thal` [2 cases]).
- **Scaling**: `StandardScaler` fitted **strictly on `X_train`**.
- **PCA for QML**: 4 and 6 components fitted **strictly on `X_train`** (capturing 55.7% and 70.1% cumulative variance).

---

## 5. Benchmark Performance Results (Untouched Test Set)

```
Model                  | Acc    | BalAcc | Prec   | Rec    | MacF1  | ROC-AUC | Time (s)
--------------------------------------------------------------------------------------------
HistGradientBoosting   | 0.9180 | 0.9215 | 0.8710 | 0.9643 | 0.9179 | 0.9610 |   0.0741
Random Forest          | 0.8852 | 0.8885 | 0.8387 | 0.9286 | 0.8851 | 0.9589 |   0.0866
Logistic Regression    | 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8688 | 0.9513 |   0.0033
Linear SVM (Calibrated)| 0.8689 | 0.8734 | 0.8125 | 0.9286 | 0.8688 | 0.9502 |   0.0190
XGBoost                | 0.8525 | 0.8582 | 0.7879 | 0.9286 | 0.8525 | 0.9297 |   0.1586
QML (4 Qubits)         | 0.7705 | 0.7716 | 0.7333 | 0.7857 | 0.7699 | 0.8333 |   5.6395
QML (6 Qubits)         | 0.7377 | 0.7332 | 0.7308 | 0.6786 | 0.7342 | 0.7955 |   9.0034
```

---

## 6. Quantum Machine Learning (QML) Architecture
- **Simulator**: PennyLane `default.qubit` with PyTorch backend.
- **Feature Map**: `qml.AngleEmbedding` on 4-qubit and 6-qubit registers.
- **Variational Circuit**: `qml.StronglyEntanglingLayers(n_layers=2)`.
- **Training**: 50 epochs, Adam optimizer ($\eta = 0.02$), batch size 32.

---

## 7. How to Train the Pipeline
```bash
python HEART/run_heart_pipeline.py
```

---

## 8. How to Run Standalone Inference
```bash
# High-risk cardiac patient example:
python HEART/predict_heart.py --age 65 --sex 1 --cp 4 --trestbps 160 --chol 280 --fbs 1 --restecg 2 --thalach 110 --exang 1 --oldpeak 2.6 --slope 2 --ca 2 --thal 7 --model ml

# Quantum QML model inference:
python HEART/predict_heart.py --age 65 --sex 1 --cp 4 --trestbps 160 --chol 280 --fbs 1 --restecg 2 --thalach 110 --exang 1 --oldpeak 2.6 --slope 2 --ca 2 --thal 7 --model qml
```

---

## 9. How to Verify Reproducibility
```bash
# Run standalone verification:
python HEART/verify_reproducibility.py

# Run unit test suite:
python -m unittest discover -s HEART/tests
```

---

## 10. Limitations
1. Historical cohort collected in 1988 (lacks modern troponin assays and coronary CTA).
2. Predominantly male sample (68.0%), reflecting 1980s referral demographics.
3. Prototype intended solely for algorithmic benchmarking for Smart India Hackathon.
