# Android Software Trust

A research project and decision-support framework focused on assessing the cybersecurity risk and trustworthiness of Android software using multiple evidence sources.

---

## Overview

Traditional Android malware detection relies heavily on isolated single-source analysis:
- **Static Evidence:** Permissions, intent filters, API calls (vulnerable to bytecode obfuscation and dynamic class loading).
- **Behavioural Evidence:** Runtime system call execution and process telemetry (vulnerable to sandbox detection or delayed triggers).
- **Network Evidence:** Network traffic flows, port communications, and packet statistics (blind to offline malware).

**Android Software Trust** explores a multi-source evidence fusion architecture that synthesizes these three pillars:
1. **Static Evidence (DREBIN Dataset)** — 214 static binary features.
2. **Behavioural Evidence (CICMalDroid 2020)** — 470 system call features.
3. **Network Evidence (CIC-AndMal2017)** — 72 statistical network flow features.

The fusion layer computes:
- **Overall Suspiciousness Risk:** Weighted aggregation of available source probabilities ($[0, 1]$).
- **Cross-Source Disagreement:** Weighted variance across sources ($[0, 0.25]$).
- **Evidence Coverage:** Proportion of total evidence weight available when sources are missing ($[0, 1]$).
- **Trust-Oriented Score:** Benign-oriented confidence heuristic penalized by risk, disagreement, and missing evidence ($[0, 1]$).
- **Interpretable Decision State:** Three-tier classification (**Trusted**, **Review**, **High Risk**).

---

## Project Structure

```text
Android-Software-Trust/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Project dependencies
├── pytest.ini                  # Pytest configuration
├── docs/
│   └── IMPLEMENTATION_NOTE.md  # Research audit & mathematical inconsistencies document
├── src/
│   ├── config.py               # Central configuration, defaults, and research presets
│   ├── fusion.py               # Tested evidence fusion and trust mathematical engine
│   ├── metrics.py              # Historical research metrics and ablation data loader
│   └── history.py              # Local assessment audit logging and CSV export
├── tests/
│   └── test_fusion.py          # 22 automated unit tests
├── notebooks/
│   └── 01_dataset_audit.ipynb  # Historical research notebook (Ablations 1-11)
└── data/raw/                   # Benchmark dataset archives
```

---

## Quickstart Guide (Windows PowerShell)

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```powershell
pytest
```
*Expected: 22 passed tests covering complete evidence combinations, edge cases, and numerical boundaries.*

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
*The cybersecurity dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## Web Application Features

1. **Overview Dashboard:** Executive summary, system readiness status, architecture flowchart, and active configuration cards.
2. **Evidence Sources:** Detailed cards for Static (DREBIN), Behavioural (CICMalDroid), and Network (CIC-AndMal2017) datasets with metrics and threat limitations.
3. **Risk Assessment:** Interactive manual evidence calculator supporting preset scenarios, missing source handling, per-source contribution breakdowns, and history logging.
4. **Research & Experiments:** Interactive tables and Plotly visualizations for historical results from `01_dataset_audit.ipynb` (Ablations 1–11, baselines, calibrations, bootstrap stability).
5. **Configuration:** Interactive tuning of weights and thresholds with real-time validation and research preset switching.
6. **Assessment History:** Audit trail of evaluations with instant CSV export and history clearing.
7. **About & Limitations:** Full research mission overview and complete Implementation Note viewer.

---

## Critical Scientific Disclosures

- **Research Stage:** Current capabilities represent an offline dataset audit and interactive prototype risk calculator. It does not perform live, runtime APK decompilation or dynamic execution.
- **Sample Alignment Constraint:** In historical offline experiments, test set predictions from disparate datasets were aligned by row index (first 2,351 samples) and evaluated against CICMalDroid ground truth. The rows do not represent identical APK binaries; results (such as behaviour-dominant weights achieving ~0.9531 balanced accuracy) reflect this offline experimental setup and are preserved for research transparency.
- **Host Security:** Unknown APK binaries must never be executed directly on the host machine. Future dynamic analysis requires an isolated Android sandbox.
