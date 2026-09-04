# Smart India Hackathon (SIH) Multi-Disease Classical & Quantum ML Platform

A unified, reproducible machine learning (ML), PennyLane quantum machine learning (QML), and RAG-assisted medical decision support system designed for multi-disease diagnosis and screening.

---

## 1. Project Objective
This platform provides high-accuracy disease screening algorithms by comparing classical machine learning models (XGBoost, Random Forest, HistGradientBoosting) against hybrid Variational Quantum Classifiers (VQC) powered by PennyLane and PyTorch.

---

## 2. Implemented Disease Modules
| Disease Module | Target Definition | Dataset Source | Pipeline Status | Best ML Model | Best QML Model |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **HCV (Hepatitis C)** | Healthy/Control vs HCV Pathology | UCI Dataset #571 (`hcvdat0.csv`) | **FINALIZED** | **XGBoost (98.37% Acc)** | **4-Qubit VQC (91.87% Acc)** |
| **DIABETES** | Non-Diabetic, Pre-Diabetic, Diabetic | Baghdad Medical City Hospital | **VALIDATED** | **Random Forest (98.11% Acc)** | **8-Qubit VQC (77.36% Acc)** |
| **HEART** | Cardiovascular Pathology Screening | UCI Heart Disease (#45) | **READY (In Progress)** | *See Handoff Guide* | *See Handoff Guide* |

---

## 3. HCV Module Summary & Benchmark Results

### Binary Clinical Target
- **Class 0 (Healthy/Control)**: Blood Donors & Suspect Donors (540 samples, 87.80%)
- **Class 1 (HCV-Related Pathology)**: Hepatitis, Fibrosis & Cirrhosis (75 samples, 12.20%)
- **Medical Scope**: *"Classification of healthy/control samples versus HCV-related liver pathology."*

### Benchmark Results (Untouched 123-Sample Test Set)
```
Model                  | Acc    | BalAcc | Prec   | Rec    | MacF1  | ROC-AUC | Time (s)
--------------------------------------------------------------------------------------------
XGBoost (Top ML)       | 0.9837 | 0.9333 | 1.0000 | 0.8667 | 0.9597 | 0.9963 |   0.1771
Random Forest          | 0.9756 | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.9972 |   0.1453
HistGradientBoosting   | 0.9675 | 0.8954 | 0.9231 | 0.8000 | 0.8571 | 0.9957 |   1.5883
Logistic Regression    | 0.9512 | 0.8000 | 1.0000 | 0.6000 | 0.7500 | 0.9235 |   0.0166
Linear SVM             | 0.9431 | 0.7667 | 1.0000 | 0.5333 | 0.6957 | 0.9778 |   0.0423
QML (4 Qubits)         | 0.9187 | 0.7528 | 0.7273 | 0.5333 | 0.7850 | 0.9290 |  24.5583
QML (6 Qubits)         | 0.8943 | 0.6528 | 0.6250 | 0.3333 | 0.6882 | 0.9333 |  39.1006
```

---

## 4. Repository Architecture

```
SIH/
├── README.md                          # Main project documentation
├── requirements.txt                   # Dependency environment specification
├── .gitignore                         # Version control exclusions
│
├── HCV/                               # Finalized HCV Module
│   ├── data/
│   │   ├── raw/hcvdat0.csv            # Official UCI Dataset #571
│   │   └── processed/cleaned_data/    # Train & test partitions
│   ├── src/
│   │   ├── preprocessing/             # Preprocessing & scaling module
│   │   ├── ml/                        # Classical ML training scripts
│   │   ├── qml/                       # PennyLane PyTorch VQC scripts
│   │   └── inference/predict_hcv.py   # Audited inference tool
│   ├── artifacts/
│   │   ├── models/                    # Saved joblib & pt binaries
│   │   ├── preprocessing/             # Fitted scalers & PCA
│   │   ├── metrics/                   # JSON logs & text reports
│   │   └── plots/                     # High-res publication plots
│   ├── tests/test_hcv_pipeline.py     # Automated test suite
│   ├── run_hcv_pipeline.py            # Master execution script
│   └── FINAL_REPORT.md                # 16-section technical engineering report
│
├── DIABETES/                          # Validated Diabetes Module
│   └── ...
│
├── HEART/                             # Heart Disease Module (In Development)
├── FRONTEND/                          # React / Web User Interface
├── BACKEND/                           # REST API Service Node
├── RAG/                               # Clinical Knowledge Explanation Layer
│
└── docs/                              # Project Documentation
    ├── TEAM_HANDOFF.md                # Developer guide for adding new disease modules
    ├── architecture/system_architecture.md
    ├── datasets/
    └── model_results/
```

---

## 5. Installation & Execution Guide

### Installation
```bash
git clone https://github.com/XenoN009/first-repo.git SIH
cd SIH
pip install -r requirements.txt
```

### Run Entire HCV Pipeline (Preprocessing, Training & Plots)
```bash
python HCV/run_hcv_pipeline.py
```

### Run Automated Tests & Reproducibility Suite
```bash
python -m unittest discover -s HCV/tests
```

### Patient Inference CLI Tool
```bash
python HCV/predict_hcv.py --Age 50 --Sex m --ALB 38.5 --ALP 52.5 --ALT 120 --AST 150 --BIL 45 --CHE 8.5 --CHOL 5.2 --CREA 75 --GGT 180 --PROT 72
```

---

## 6. How Team Members Should Add New Disease Modules
Refer to [`docs/TEAM_HANDOFF.md`](file:///c:/Users/HP/Desktop/SIH/docs/TEAM_HANDOFF.md) for step-by-step instructions on implementing the `HEART` module following the identical 16-step quality standard.

---

## 7. Important Medical Disclaimer & Limitations
> [!CAUTION]
> This system is an **AI/QML research prototype** built for Smart India Hackathon benchmarking. It is **NOT a clinically validated diagnostic system**. Predictions must be interpreted by qualified medical professionals alongside clinical lab confirmatory testing.
