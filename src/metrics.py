"""Research metrics and experimental data loader for Android Software Trust.

Contains the verified experimental results recorded in notebooks/01_dataset_audit.ipynb,
including baseline evaluations, model calibrations, source-quality rankings,
and Ablations 1 through 11 with sample alignment disclosures.
"""

from typing import Dict, Any, List
import pandas as pd


# ==============================================================================
# 1. DATASET AUDIT & METADATA
# ==============================================================================
DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "Static / DREBIN": {
        "dataset_name": "DREBIN Android Malware Dataset",
        "evidence_type": "Static Analysis (Manifest, Permissions, Intents, API Calls)",
        "features": 214,
        "feature_type": "Binary feature indicators",
        "total_samples": 15036,
        "test_samples": 2628,
        "split_strategy": "Feature-vector group-aware 80/20 train/test split",
        "leakage_safeguard": "Identical feature vectors placed strictly in the same split to eliminate data leakage.",
        "best_model": "Calibrated Extra Trees Classifier",
        "status": "Offline Dataset Experiment",
        "limitations": "Static signatures can be evaded via bytecode obfuscation, reflection, and dynamic class loading."
    },
    "Behavioural / CICMalDroid": {
        "dataset_name": "CICMalDroid 2020",
        "evidence_type": "Dynamic Behavioural Analysis (System Calls, Process Execution)",
        "features": 470,
        "feature_type": "System call frequencies & behavioral telemetry",
        "total_samples": 11598,
        "test_samples": 2351,
        "split_strategy": "Group-aware 80/20 split based on package/family identity",
        "leakage_safeguard": "Prevented malware families or repackaged clones from appearing in both train and test.",
        "best_model": "Calibrated Random Forest Classifier",
        "status": "Offline Dataset Experiment",
        "limitations": "Requires live execution in sandbox; evasion via sandbox detection, sleep calls, or delayed payload triggering."
    },
    "Network / CIC-AndMal": {
        "dataset_name": "CIC-AndMal2017",
        "evidence_type": "Network Traffic Telemetry (Packet Flow, Durations, Ports, Bytes)",
        "features": 72,
        "feature_type": "Continuous network flow statistical features (12 constant features removed)",
        "total_samples": 426733,
        "test_samples": 85347,
        "split_strategy": "Standard temporal / stratified 80/20 flow split",
        "leakage_safeguard": "Eliminated 12 invariant constant features with zero variance prior to training.",
        "best_model": "Calibrated Random Forest Classifier",
        "status": "Offline Dataset Experiment",
        "limitations": "Captures only network flows active during testing; offline malware or non-network malicious payloads are invisible."
    }
}


# ==============================================================================
# 2. MODEL COMPARISON & CALIBRATION (Notebook Cell 180)
# ==============================================================================
MODEL_COMPARISON_DATA: List[Dict[str, Any]] = [
    {
        "Evidence": "Static / DREBIN",
        "Model": "Random Forest",
        "F1": 0.96863,
        "Brier": 0.025048,
        "ECE": 0.047846,
        "Calibrated_Brier": 0.018768,
        "Calibrated_ECE": 0.016180,
        "Selected": "No"
    },
    {
        "Evidence": "Static / DREBIN",
        "Model": "Extra Trees",
        "F1": 0.97425,
        "Brier": 0.022477,
        "ECE": 0.041688,
        "Calibrated_Brier": 0.016965,
        "Calibrated_ECE": 0.009279,
        "Selected": "YES"
    },
    {
        "Evidence": "Static / DREBIN",
        "Model": "Logistic Regression",
        "F1": 0.95563,
        "Brier": 0.027173,
        "ECE": 0.018879,
        "Calibrated_Brier": None,
        "Calibrated_ECE": None,
        "Selected": "No"
    },
    {
        "Evidence": "Static / DREBIN",
        "Model": "HistGradientBoosting",
        "F1": 0.96668,
        "Brier": 0.019866,
        "ECE": 0.012817,
        "Calibrated_Brier": 0.019271,
        "Calibrated_ECE": 0.013590,
        "Selected": "No"
    },
    {
        "Evidence": "Behaviour / CICMalDroid",
        "Model": "Random Forest",
        "F1": 0.98686,
        "Brier": 0.021542,
        "ECE": 0.025191,
        "Calibrated_Brier": 0.018844,
        "Calibrated_ECE": 0.014311,
        "Selected": "YES"
    },
    {
        "Evidence": "Behaviour / CICMalDroid",
        "Model": "Extra Trees",
        "F1": 0.92883,
        "Brier": 0.023761,
        "ECE": 0.026349,
        "Calibrated_Brier": None,
        "Calibrated_ECE": None,
        "Selected": "No"
    },
    {
        "Evidence": "Network / CIC-AndMal",
        "Model": "Random Forest",
        "F1": 0.97594,
        "Brier": 0.013643,
        "ECE": 0.004147,
        "Calibrated_Brier": 0.014239,
        "Calibrated_ECE": 0.003618,
        "Selected": "YES"
    },
    {
        "Evidence": "Network / CIC-AndMal",
        "Model": "Extra Trees",
        "F1": 0.97370,
        "Brier": 0.014566,
        "ECE": 0.004008,
        "Calibrated_Brier": None,
        "Calibrated_ECE": None,
        "Selected": "No"
    }
]


# ==============================================================================
# 3. EVIDENCE RELIABILITY WEIGHT DERIVATION (Notebook Cell 183)
# ==============================================================================
RELIABILITY_WEIGHT_DATA: List[Dict[str, Any]] = [
    {
        "Evidence": "Static / DREBIN",
        "F1": 0.9749,
        "Brier": 0.0170,
        "ECE": 0.0093,
        "Brier_Reliability": 58.9434,
        "ECE_Reliability": 107.7751,
        "F1_Norm": 0.3319,
        "Brier_Norm": 0.3181,
        "ECE_Norm": 0.2573,
        "Composite_Weight": 0.3024
    },
    {
        "Evidence": "Behaviour / CICMalDroid",
        "F1": 0.9869,
        "Brier": 0.0188,
        "ECE": 0.0143,
        "Brier_Reliability": 53.0663,
        "ECE_Reliability": 69.8746,
        "F1_Norm": 0.3359,
        "Brier_Norm": 0.2864,
        "ECE_Norm": 0.1668,
        "Composite_Weight": 0.2630
    },
    {
        "Evidence": "Network / CIC-AndMal",
        "F1": 0.9759,
        "Brier": 0.0136,
        "ECE": 0.0041,
        "Brier_Reliability": 73.2975,
        "ECE_Reliability": 241.1568,
        "F1_Norm": 0.3322,
        "Brier_Norm": 0.3955,
        "ECE_Norm": 0.5758,
        "Composite_Weight": 0.4345
    }
]


# ==============================================================================
# 4. ABLATION STUDIES 1 - 11 (From Notebook Cells 200 - 213)
# ==============================================================================

# Ablation 1: Decision Distribution (Notebook Cell 200)
ABLATION_1_DATA: List[Dict[str, Any]] = [
    {
        "Method": "Equal Weights [0.33, 0.33, 0.33]",
        "Mean_Risk": 0.5224,
        "Mean_Disagreement": 0.1636,
        "Mean_Trust": 0.4397,
        "Trusted_%": 0.0,
        "Review_%": 49.09,
        "High_Risk_%": 50.91,
    },
    {
        "Method": "Reliability Weights [0.25, 0.27, 0.48]",
        "Mean_Risk": 0.4829,
        "Mean_Disagreement": 0.1589,
        "Mean_Trust": 0.4101,
        "Trusted_%": 0.0,
        "Review_%": 72.39,
        "High_Risk_%": 27.61,
    }
]

# Ablation 2: Ground-Truth Validation on CICMalDroid Test Labels (Notebook Cell 203)
ABLATION_2_DATA: List[Dict[str, Any]] = [
    {
        "Method": "Equal Weights",
        "Weights": "[0.333, 0.333, 0.333]",
        "Accuracy": 0.5364,
        "Balanced_Accuracy": 0.7158,
        "F1": 0.6282,
        "Benign_Correct": 340,
        "Benign_Total": 350,
        "Malware_Correct": 921,
        "Malware_Total": 2001
    },
    {
        "Method": "Reliability Weights",
        "Weights": "[0.250, 0.269, 0.481]",
        "Accuracy": 0.1519,
        "Balanced_Accuracy": 0.4805,
        "F1": 0.0245,
        "Benign_Correct": 332,
        "Benign_Total": 350,
        "Malware_Correct": 25,
        "Malware_Total": 2001
    }
]

# Ablation 4: Weight Sensitivity Analysis (Notebook Cell 205)
ABLATION_4_DATA: List[Dict[str, Any]] = [
    {"Strategy": "Behaviour Dominant", "Static_W": 0.20, "Behaviour_W": 0.60, "Network_W": 0.20, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Disagreement": 0.1421},
    {"Strategy": "Behaviour Heavy (0.50)", "Static_W": 0.25, "Behaviour_W": 0.50, "Network_W": 0.25, "Accuracy": 0.6891, "Balanced_Accuracy": 0.8067, "F1": 0.7777, "Disagreement": 0.1364},
    {"Strategy": "Equal Weights", "Static_W": 0.333, "Behaviour_W": 0.333, "Network_W": 0.333, "Accuracy": 0.6712, "Balanced_Accuracy": 0.7892, "F1": 0.7628, "Disagreement": 0.1636},
    {"Strategy": "Reliability Weights", "Static_W": 0.250, "Behaviour_W": 0.269, "Network_W": 0.481, "Accuracy": 0.6342, "Balanced_Accuracy": 0.7391, "F1": 0.7329, "Disagreement": 0.1589},
    {"Strategy": "Static Dominant", "Static_W": 0.60, "Behaviour_W": 0.20, "Network_W": 0.20, "Accuracy": 0.5368, "Balanced_Accuracy": 0.7255, "F1": 0.6267, "Disagreement": 0.1292},
    {"Strategy": "Network Dominant", "Static_W": 0.20, "Behaviour_W": 0.20, "Network_W": 0.60, "Accuracy": 0.3713, "Balanced_Accuracy": 0.4892, "F1": 0.4653, "Disagreement": 0.1411},
]

# Ablation 5: Source Removal & Evidence Contribution (Notebook Cell 206)
ABLATION_5_DATA: List[Dict[str, Any]] = [
    {"Configuration": "Behaviour Only", "Sources_Count": 1, "Weights": "[1.00]", "Accuracy": 0.9749, "Balanced_Accuracy": 0.9381, "F1": 0.9853, "Disagreement": 0.0000},
    {"Configuration": "All Three (Equal)", "Sources_Count": 3, "Weights": "[0.33, 0.33, 0.33]", "Accuracy": 0.6712, "Balanced_Accuracy": 0.7892, "F1": 0.7628, "Disagreement": 0.1636},
    {"Configuration": "Static + Behaviour", "Sources_Count": 2, "Weights": "[0.50, 0.50]", "Accuracy": 0.6287, "Balanced_Accuracy": 0.7771, "F1": 0.7217, "Disagreement": 0.1114},
    {"Configuration": "Static Only", "Sources_Count": 1, "Weights": "[1.00]", "Accuracy": 0.5321, "Balanced_Accuracy": 0.7216, "F1": 0.6217, "Disagreement": 0.0000},
    {"Configuration": "Static + Network", "Sources_Count": 2, "Weights": "[0.50, 0.50]", "Accuracy": 0.5151, "Balanced_Accuracy": 0.6597, "F1": 0.6143, "Disagreement": 0.1083},
    {"Configuration": "Behaviour + Network", "Sources_Count": 2, "Weights": "[0.50, 0.50]", "Accuracy": 0.3998, "Balanced_Accuracy": 0.5142, "F1": 0.4991, "Disagreement": 0.1485},
    {"Configuration": "Network Only", "Sources_Count": 1, "Weights": "[1.00]", "Accuracy": 0.3688, "Balanced_Accuracy": 0.4842, "F1": 0.4631, "Disagreement": 0.0000},
]

# Ablation 6: Weighting Strategy Comparison (Notebook Cell 207)
ABLATION_6_DATA: List[Dict[str, Any]] = [
    {"Method": "Behaviour Dominant", "Static_W": 0.2000, "Behaviour_W": 0.6000, "Network_W": 0.2000, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Mean_Disagreement": 0.1421, "Mean_Trust": 0.5604},
    {"Method": "Equal", "Static_W": 0.3333, "Behaviour_W": 0.3333, "Network_W": 0.3333, "Accuracy": 0.6712, "Balanced_Accuracy": 0.7892, "F1": 0.7628, "Mean_Disagreement": 0.1636, "Mean_Trust": 0.4402},
    {"Method": "F1 Derived", "Static_W": 0.3304, "Behaviour_W": 0.3366, "Network_W": 0.3329, "Accuracy": 0.6708, "Balanced_Accuracy": 0.7889, "F1": 0.7624, "Mean_Disagreement": 0.1638, "Mean_Trust": 0.4414},
    {"Method": "Brier Derived", "Static_W": 0.2501, "Behaviour_W": 0.2908, "Network_W": 0.4591, "Accuracy": 0.6487, "Balanced_Accuracy": 0.7630, "F1": 0.7441, "Mean_Disagreement": 0.1614, "Mean_Trust": 0.4191},
    {"Method": "Reliability Derived", "Static_W": 0.2500, "Behaviour_W": 0.2693, "Network_W": 0.4807, "Accuracy": 0.6342, "Balanced_Accuracy": 0.7391, "F1": 0.7329, "Mean_Disagreement": 0.1589, "Mean_Trust": 0.4107},
    {"Method": "ECE Derived", "Static_W": 0.0693, "Behaviour_W": 0.1316, "Network_W": 0.7992, "Accuracy": 0.3692, "Balanced_Accuracy": 0.4845, "F1": 0.4637, "Mean_Disagreement": 0.0905, "Mean_Trust": 0.3722},
]

# Ablation 9: Bootstrap Resampling Robustness (Notebook Cell 211)
ABLATION_9_DATA: List[Dict[str, Any]] = [
    {"Strategy": "Behaviour Dominant [0.2, 0.6, 0.2]", "Full_Dataset_Bal_Acc": 0.9531, "Bootstrap_Mean_Bal_Acc": 0.9531, "Full_Dataset_F1": 0.9873, "Bootstrap_Mean_F1": 0.9873, "Stability": "High (Resampling invariant)"},
    {"Strategy": "Equal Weights [0.33, 0.33, 0.33]", "Full_Dataset_Bal_Acc": 0.7892, "Bootstrap_Mean_Bal_Acc": 0.7890, "Full_Dataset_F1": 0.7628, "Bootstrap_Mean_F1": 0.7626, "Stability": "Moderate"},
    {"Strategy": "Reliability Weights [0.25, 0.27, 0.48]", "Full_Dataset_Bal_Acc": 0.7391, "Bootstrap_Mean_Bal_Acc": 0.7392, "Full_Dataset_F1": 0.7329, "Bootstrap_Mean_F1": 0.7328, "Stability": "Moderate"},
]

# Ablation 10: Threshold Sensitivity on Behaviour Dominant Weights (Notebook Cell 212)
ABLATION_10_DATA: List[Dict[str, Any]] = [
    {"Trusted_T": 0.10, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 7.32, "Review_%": 7.27, "High_Risk_%": 85.41},
    {"Trusted_T": 0.15, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 7.74, "Review_%": 6.85, "High_Risk_%": 85.41},
    {"Trusted_T": 0.20, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 8.42, "Review_%": 6.17, "High_Risk_%": 85.41},
    {"Trusted_T": 0.25, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 9.27, "Review_%": 5.32, "High_Risk_%": 85.41},
    {"Trusted_T": 0.30, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 10.42, "Review_%": 4.17, "High_Risk_%": 85.41},
    {"Trusted_T": 0.35, "HighRisk_T": 0.50, "Accuracy": 0.9813, "Balanced_Accuracy": 0.9608, "F1": 0.9873, "Trusted_%": 11.23, "Review_%": 3.36, "High_Risk_%": 85.41},
    {"Trusted_T": 0.40, "HighRisk_T": 0.50, "Accuracy": 0.9783, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Trusted_%": 12.12, "Review_%": 2.47, "High_Risk_%": 85.41},
]

# Ablation 11: Principled Source Quality Weighting (Notebook Cell 213)
ABLATION_11_DATA: List[Dict[str, Any]] = [
    {"Method": "Behaviour Dominant", "Static_W": 0.2000, "Behaviour_W": 0.6000, "Network_W": 0.2000, "Balanced_Accuracy": 0.9531, "F1": 0.9873, "Accuracy": 0.9783, "Disagreement": 0.1421, "Trust": 0.5604},
    {"Method": "F1 Derived", "Static_W": 0.3003, "Behaviour_W": 0.4760, "Network_W": 0.2237, "Balanced_Accuracy": 0.8014, "F1": 0.7811, "Accuracy": 0.6920, "Disagreement": 0.1560, "Trust": 0.5047},
    {"Method": "Balanced Accuracy Derived", "Static_W": 0.3366, "Behaviour_W": 0.4376, "Network_W": 0.2259, "Balanced_Accuracy": 0.8004, "F1": 0.7741, "Accuracy": 0.6844, "Disagreement": 0.1572, "Trust": 0.4898},
    {"Method": "Principled Quality", "Static_W": 0.3151, "Behaviour_W": 0.4512, "Network_W": 0.2338, "Balanced_Accuracy": 0.7988, "F1": 0.7754, "Accuracy": 0.6857, "Disagreement": 0.1579, "Trust": 0.4939},
    {"Method": "Equal", "Static_W": 0.3333, "Behaviour_W": 0.3333, "Network_W": 0.3333, "Balanced_Accuracy": 0.7892, "F1": 0.7628, "Accuracy": 0.6712, "Disagreement": 0.1636, "Trust": 0.4402},
    {"Method": "Reliability", "Static_W": 0.2500, "Behaviour_W": 0.2693, "Network_W": 0.4807, "Balanced_Accuracy": 0.7391, "F1": 0.7329, "Accuracy": 0.6342, "Disagreement": 0.1589, "Trust": 0.4107},
]


# ==============================================================================
# DATAFRAME GENERATORS FOR UI & CHARTS
# ==============================================================================
def get_model_comparison_df() -> pd.DataFrame:
    """Return model comparison dataframe."""
    return pd.DataFrame(MODEL_COMPARISON_DATA)


def get_reliability_weights_df() -> pd.DataFrame:
    """Return reliability weight breakdown dataframe."""
    return pd.DataFrame(RELIABILITY_WEIGHT_DATA)


def get_ablation_df(ablation_num: int) -> pd.DataFrame:
    """Return the dataframe for a specific ablation study."""
    mapping = {
        1: ABLATION_1_DATA,
        2: ABLATION_2_DATA,
        4: ABLATION_4_DATA,
        5: ABLATION_5_DATA,
        6: ABLATION_6_DATA,
        9: ABLATION_9_DATA,
        10: ABLATION_10_DATA,
        11: ABLATION_11_DATA,
    }
    if ablation_num not in mapping:
        raise ValueError(f"Ablation {ablation_num} data not available. Supported: {list(mapping.keys())}")
    return pd.DataFrame(mapping[ablation_num])


# ==============================================================================
# 5. RESEARCH PAPER SPECIFIC TABLES & CITATION DATA
# ==============================================================================
PAPER_METADATA = {
    "title": "Source-Aware Evidence Fusion for Multi-Source Security Risk Assessment",
    "authors": [
        "Paramjeet Kaur",
        "Priya Goel",
        "Vrinda Sachdeva",
        "Avneesh Kumar",
        "Janvi Chaudhary",
        "Mahek Singhal"
    ],
    "institution": "Department of Computer Science and Engineering, GL Bajaj Institute of Technology and Management, Greater Noida, India",
    "central_question": "When heterogeneous security sources differ in predictive quality, how does source contribution affect the quality of the final security decision?",
    "key_findings": {
        "eq18_recovery": "0.9531 - 0.7892 = 0.1639 (+16.39 percentage-point BA improvement over equal fusion)",
        "eq19_modest_gain": "0.9531 - 0.9381 = 0.0150 (+1.50 point gain over best individual source - behavioural)",
        "eq20_bootstrap": "0.9531 ± 0.0075 (95% CI: [0.9382, 0.9674], P(BD > Equal) = 1.000)",
        "synthetic_disagreement": "10,000 synthetic cases tested: 254 exhibited low risk + high disagreement (D >= 0.10). 100% routed to Review, 0 to Trusted."
    }
}

PAPER_TABLE_I_DATA = [
    {"Source": "Static (DREBIN)", "Balanced_Accuracy": 0.7216, "F1": 0.6217, "ECE": 0.4575},
    {"Source": "Behavioural (CICMalDroid)", "Balanced_Accuracy": 0.9381, "F1": 0.9853, "ECE": 0.0143},
    {"Source": "Network (CIC-AndMal)", "Balanced_Accuracy": 0.4842, "F1": 0.4631, "ECE": 0.6215},
]

PAPER_TABLE_II_DATA = [
    {"Configuration": "Behaviour only", "Balanced_Accuracy": 0.9381, "F1": 0.9853},
    {"Configuration": "Static only", "Balanced_Accuracy": 0.7216, "F1": 0.6217},
    {"Configuration": "Network only", "Balanced_Accuracy": 0.4842, "F1": 0.4631},
    {"Configuration": "Static + Behaviour", "Balanced_Accuracy": 0.7771, "F1": 0.7217},
    {"Configuration": "Static + Network", "Balanced_Accuracy": 0.6597, "F1": 0.6143},
    {"Configuration": "Behaviour + Network", "Balanced_Accuracy": 0.5142, "F1": 0.4991},
    {"Configuration": "All three, equal", "Balanced_Accuracy": 0.7892, "F1": 0.7628},
]

PAPER_TABLE_III_DATA = [
    {"Strategy": "Equal", "ws": 0.333, "wb": 0.333, "wn": 0.333, "Balanced_Accuracy": 0.7892, "F1": 0.7628},
    {"Strategy": "F1-derived", "ws": 0.330, "wb": 0.337, "wn": 0.333, "Balanced_Accuracy": 0.7889, "F1": 0.7624},
    {"Strategy": "Brier-derived", "ws": 0.250, "wb": 0.291, "wn": 0.459, "Balanced_Accuracy": 0.7630, "F1": 0.7441},
    {"Strategy": "ECE-derived", "ws": 0.069, "wb": 0.132, "wn": 0.799, "Balanced_Accuracy": 0.4845, "F1": 0.4637},
    {"Strategy": "Reliability", "ws": 0.319, "wb": 0.325, "wn": 0.356, "Balanced_Accuracy": 0.7391, "F1": 0.7329},
    {"Strategy": "Behaviour-Dominant", "ws": 0.200, "wb": 0.600, "wn": 0.200, "Balanced_Accuracy": 0.9531, "F1": 0.9873},
]

PAPER_TABLE_IV_DATA = [
    {"Source": "Static (DREBIN)", "F1": 0.97495, "Cal_Brier": 0.01697, "Cal_ECE": 0.00928},
    {"Source": "Behavioural (CICMalDroid)", "F1": 0.98686, "Cal_Brier": 0.01884, "Cal_ECE": 0.01431},
    {"Source": "Network (CIC-AndMal)", "F1": 0.97594, "Cal_Brier": 0.01424, "Cal_ECE": 0.00362},
]

PAPER_FIG1_DATA = [
    {"Configuration": "B (Behavioural)", "Category": "Individual", "Balanced_Accuracy": 0.9381, "Label": "0.94"},
    {"Configuration": "S (Static)", "Category": "Individual", "Balanced_Accuracy": 0.7216, "Label": "0.72"},
    {"Configuration": "N (Network)", "Category": "Individual", "Balanced_Accuracy": 0.4842, "Label": "0.48"},
    {"Configuration": "S+B", "Category": "Pairwise", "Balanced_Accuracy": 0.7771, "Label": "0.78"},
    {"Configuration": "S+N", "Category": "Pairwise", "Balanced_Accuracy": 0.6597, "Label": "0.66"},
    {"Configuration": "B+N", "Category": "Pairwise", "Balanced_Accuracy": 0.5142, "Label": "0.51"},
    {"Configuration": "All (Equal)", "Category": "Equal Fusion", "Balanced_Accuracy": 0.7892, "Label": "0.79"},
    {"Configuration": "BD (Behaviour-Dominant)", "Category": "Proposed", "Balanced_Accuracy": 0.9531, "Label": "0.95"},
]


def get_paper_metadata() -> Dict[str, Any]:
    return PAPER_METADATA

def get_paper_table1_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_TABLE_I_DATA)

def get_paper_table2_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_TABLE_II_DATA)

def get_paper_table3_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_TABLE_III_DATA)

def get_paper_table4_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_TABLE_IV_DATA)

def get_paper_fig1_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_FIG1_DATA)
