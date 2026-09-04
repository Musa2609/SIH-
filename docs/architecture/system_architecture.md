# SIH System Architecture Specification
## Multi-Disease Quantum & Classical ML Clinical Decision Support System

This document specifies the technical architecture of the Smart India Hackathon (SIH) Clinical Decision Support System. The platform combines classical machine learning (ML), PennyLane quantum machine learning (QML), and a Retrieval-Augmented Generation (RAG) explanation layer.

---

## 1. High-Level System Architecture & Flow

```mermaid
flowchart TD
    subgraph Client Layer
        A[User / Clinician Interface] -->|Submits Biomarkers| B[Frontend UI]
    end

    subgraph API & Gateway Layer
        B -->|REST / JSON API| C[Backend Gateway Node]
        C --> D{Input Validation Engine}
        D -->|Invalid Schema| E[Validation Error Response]
        D -->|Valid Features| F[Disease Router]
    end

    subgraph Disease Execution Engines
        F -->|HCV Data| G1[HCV Preprocessing Pipeline]
        F -->|Diabetes Data| G2[Diabetes Preprocessing Pipeline]
        F -->|Heart Data| G3[Heart Preprocessing Pipeline]

        subgraph Classical ML Pipeline
            G1 -->|Standardized Features| H1[XGBoost / Random Forest Classifier]
            H1 -->|Probability & Class| I[Prediction Output Engine]
        end

        subgraph Quantum ML Pipeline
            G1 -->|PCA Dimension Reduction| H2[PCA 4/6 Components]
            H2 -->|AngleEmbedding| H3[PennyLane Quantum Simulator]
            H3 -->|StronglyEntanglingLayers| H4[Quantum Readout & Logits]
            H4 -->|Quantum Class Probability| I
        end
    end

    subgraph Knowledge & RAG Explanation Layer
        I -->|Prediction + Biomarker Vector| J[RAG Retriever Node]
        J -->|Vector Similarity Search| K[(Medical Knowledge Base)]
        K -->|Clinical Context & Guidelines| L[LLM Explanation Generator]
        L -->|Patient-Friendly Report| M[Final Clinical Result & Explanation]
    end

    M -->|JSON Response| B
```

---

## 2. Pipeline Stage Breakdown

### Stage 1: User & Interface Layer
- Patients or clinical staff input biological metrics via modern web UI.
- All input fields enforce standard medical range validation (e.g., age, enzyme concentrations in U/L or mmol/L).

### Stage 2: Backend Gateway & Router
- Standardizes incoming requests and routes them to the appropriate disease-specific module (`HCV`, `DIABETES`, `HEART`).

### Stage 3: Disease-Specific Preprocessing
- Performs missing value imputation, categorical encoding, and Z-score standardization.
- Preprocessing objects (`Imputer`, `Encoder`, `Scaler`, `PCA`) are loaded from frozen production artifacts fitted strictly during offline training.

### Stage 4: Dual Execution Engine (Classical ML vs QML)
- **Classical ML Engine**: Passes full clinical feature vector through top tree ensembles (e.g., XGBoost) for high-accuracy probability scoring.
- **Quantum ML Engine**: Passes compressed PCA vectors through PennyLane variational quantum circuits (`AngleEmbedding` + `StronglyEntanglingLayers`) to provide quantum decision support.

---

## 3. RAG Explanation Layer Integration

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL BOUNDARY**: The RAG (Retrieval-Augmented Generation) layer is strictly an **explanation and patient education layer**. It **DOES NOT MODIFY, OVERRIDE, OR RETRAIN** the underlying ML or QML predictions.

### RAG Data Flow:
1. **Trigger**: ML/QML engine emits diagnostic category (e.g., `HCV-related pathology`) and confidence score (e.g., `98.74%`).
2. **Retrieval**: The system queries a vector database containing clinical guidelines (e.g., AASLD / EASL Hepatitis C management protocols) using key biomarker anomalies (e.g., elevated `AST` and `ALT`).
3. **Generation**: The RAG layer synthesizes a readable, patient-friendly summary explaining *why* the prediction was made and recommending next clinical steps (e.g., "Consult a hepatologist for confirmatory PCR testing").
