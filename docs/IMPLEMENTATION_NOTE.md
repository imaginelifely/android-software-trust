# Implementation Note: Evidence Fusion & Experimental Validity Audit

**Project:** Android Software Trust  
**Author:** Research Engineering Team  
**Date:** October 2026  
**Status:** Prototype & Experimental Research Audit  

---

## 1. Executive Summary

This document audits the mathematical formulations, configuration inconsistencies, and experimental methodology in the `Android Software Trust` codebase (specifically comparing `src/config.py`, `src/fusion.py`, and `notebooks/01_dataset_audit.ipynb`). It explains the mathematical choices adopted in the production prototype, discloses sample alignment constraints, and defines architectural boundaries for future APK runtime analysis.

---

## 2. Inconsistencies Audited & Resolution

### A. Evidence Weight Configurations

Across the codebase and notebook experiments, several distinct weighting schemes were explored:

| Configuration Scheme | Static Weight ($w_s$) | Behaviour Weight ($w_b$) | Network Weight ($w_n$) | Origin / Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Behaviour Dominant (Default)** | **0.20** | **0.60** | **0.20** | Defined in `src/config.py`. Achieved top balanced accuracy (0.9531) in notebook Ablations 4, 6, 8, 9, 10, and 11. |
| **Equal Weights** | 0.3333 | 0.3333 | 0.3333 | Standard baseline uninformative prior tested in Ablations 1, 2, 4, 5, and 6. |
| **Composite Reliability** | 0.3024 | 0.2630 | 0.4345 | Derived in Notebook Cell 184 via harmonic weighting of F1, inverse Brier score, and inverse ECE. |
| **Bounded Reliability ($\lambda=10$)** | 0.2500 | 0.2693 | 0.4807 | Notebook Cell 200 & Ablation 1. Gives highest weight to Network due to its low test ECE (0.0041). |
| **Principled Source-Quality** | 0.3151 | 0.4512 | 0.2338 | Ablation 11 multi-objective compromise (diagnostic accuracy, calibration, and behavioral relevance). |
| **Balanced Accuracy Derived** | 0.3366 | 0.4376 | 0.2259 | Ablation 8 based on single-source balanced accuracy rankings. |

**Resolution in Prototype:**
- We preserve the existing values in `src/config.py` (`[0.20, 0.60, 0.20]`) as the default configuration.
- We implement full user configurability in `src/config.py` and the UI, allowing analysts to select any named research preset or customize weights with automated validation ($\sum w_i = 1.0, w_i \ge 0$).

---

### B. Trust Score Formulations and Directionality

In the exploratory notebook, multiple trust formulas were tested:

1. **Early Heuristic (Notebook Cells 97–103):**  
   $\text{evidence\_to\_trust}(p) = 1.0 - p$  
   Fused using equal weights, penalized by variance: $\text{trust} = \text{overall\_trust} \times (1 - \text{disagreement})$.

2. **Coverage-Scaled Benign Trust (Notebook Cells 153, 161–162):**  
   $\text{base\_trust} = 1 - \text{risk}$  
   $\text{trust} = (1 - \text{risk}) \times (1 - \text{disagreement}) \times \sqrt{\text{coverage}}$  
   Here, trust represents **benign-oriented confidence**: 1.0 indicates maximum confidence that software is benign, while low trust occurs when software is risky, evidence conflicts, or sources are missing.

3. **Inconsistent Variable Naming (Notebook Cell 213 / Ablation 11):**  
   Cell 213 defined `trust = risk * (1.0 - disagreement)`. In this single cell, "trust" scaled in the same direction as risk (higher risk produced higher "trust").

**Resolution in Prototype:**
- Trust is strictly treated as a **heuristic decision-support score**, not a calibrated probability of safety.
- **Directionality is made explicit:**
  - **Risk ($[0, 1]$):** Higher score = **more suspicious / malicious**.
  - **Disagreement ($[0, 0.25]$):** Higher score = **greater source conflict**.
  - **Coverage ($[0, 1]$):** Higher score = **more complete evidence**.
  - **Trust ($[0, 1]$):** Higher score = **greater benign confidence** (penalized by risk, disagreement, and missing evidence).
- Adopted formula:
  $$\text{trust} = (1 - \text{risk}) \times (1 - \text{disagreement}) \times \sqrt{\text{coverage}}$$

---

### C. Decision Thresholds and Logic

Two thresholding paradigms appeared in the notebook:
1. **Trust-Based Thresholding (Cell 106, 163):** $\text{trust} \ge 0.70 \to \text{Trusted}$, $\text{trust} \ge 0.40 \to \text{Review}$, else $\text{High Risk}$.
2. **Risk-Based Thresholding (Cell 204, 212, `src/config.py`):**
   - $\text{risk} < 0.20 \to \text{Trusted}$
   - $\text{risk} \ge 0.50 \to \text{High Risk}$
   - $0.20 \le \text{risk} < 0.50 \to \text{Review}$

**Resolution in Prototype:**
- `src/config.py` defines `TRUSTED_THRESHOLD = 0.20` and `HIGH_RISK_THRESHOLD = 0.50`. These are directly adopted as risk thresholds.
- When $\text{risk} < T_{\text{trusted}}$ (e.g. 0.20), the decision is **Trusted**.
- When $\text{risk} \ge T_{\text{high\_risk}}$ (e.g. 0.50), the decision is **High Risk**.
- Between the thresholds, the decision is **Review**.
- Additionally, when $\text{disagreement} \ge 0.10$, an explicit **High Disagreement Alert** is triggered to warn analysts that evidence sources conflict.

---

## 3. Critical Experimental Validity & Sample Alignment Disclosure

In the offline research notebook:
- DREBIN test set contains **2,628** predictions.
- CICMalDroid 2020 test set contains **2,351** predictions.
- CIC-AndMal2017 test set contains **85,347** predictions.

To perform offline fusion experiments (Ablations 1–11), the notebook constructed an aligned matrix of shape `(2351, 3)` by taking the first 2,351 rows from each test set and evaluating them against the ground truth labels of the CICMalDroid test set (`y_cic_test_binary`: 350 benign, 2,001 malware).

### Essential Scientific Disclosure:
1. **No Shared APK Identity:** Rows 0–2,350 of DREBIN and CIC-AndMal2017 do not represent the same Android applications as rows 0–2,350 of CICMalDroid.
2. **Why Behaviour-Dominant Weights Outperformed:** Because the aligned matrix was evaluated against CICMalDroid ground truth, assigning 60% weight to CICMalDroid (the behavioural model) naturally yielded the highest metrics ($\approx 0.9531$ balanced accuracy, $0.9873$ F1). Other sources, having uncoordinated labels, acted as noise when weighted heavily.
3. **Historical Context:** These metrics are historical artifacts of an offline synthetic alignment experiment. They **do not prove multimodal real-world performance** on arbitrary APKs.

---

## 4. Current Repository Artifact Inventory

An inspection of the workspace confirms:
- **Trained Model Files:** None exist on disk (`.pkl`, `.joblib`, `.pt`, `.onnx` are absent). All models in the notebook were fitted in memory.
- **Feature Extraction Scripts:** The repository contains raw dataset CSVs (`data/raw/`), but does not contain decompilation or dynamic extraction pipelines for unknown `.apk` files.
- **Runtime APK Capability:** The repository is currently a **research dataset audit and prototype risk calculator**. It does not perform live APK analysis.

---

## 5. Architectural Boundary for Future APK Ingestion

To transition from this prototype to live APK evaluation, future development should follow this isolated pipeline:

```
[Uploaded APK]
      │
      ▼
1. Validation & Unpacking (Apktool / Androguard) ──> Static Features (DREBIN schema)
      │                                                     │
      ▼ (Isolated Android Emulator Sandbox)                 ▼
2. Dynamic Instrumentation (Frida / DroidBox) ────> Behaviour Features (CICMalDroid schema)
      │                                                     │
      ▼ (Network Interface Capture / PCAP)                  ▼
3. Flow Feature Extraction (CICFlowMeter) ────────> Network Features (CIC-AndMal schema)
      │                                                     │
      └─────────────────────┬───────────────────────────────┘
                            ▼
      4. Calibrated Base Model Inferences (P_static, P_beh, P_net)
                            │
                            ▼
      5. Evidence Fusion Engine (`src/fusion.py`)
                            │
                            ▼
      6. Explainable Decision (Risk, Trust, Disagreement, Coverage)
```
