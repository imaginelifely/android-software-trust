"""Configuration module for Android Software Trust.

Preserves the original research defaults while providing structured,
validated configuration objects and presets from historical experiments
and the published research paper.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import math

# ==============================================================================
# ORIGINAL RESEARCH CONFIGURATION DEFAULTS
# ==============================================================================
STATIC_WEIGHT: float = 0.20
BEHAVIOUR_WEIGHT: float = 0.60
NETWORK_WEIGHT: float = 0.20

# Risk-based decision thresholds
TRUSTED_THRESHOLD: float = 0.20
HIGH_RISK_THRESHOLD: float = 0.50

# Paper Section III-D Trust Decision Thresholds (Equations 10, 11, 12)
PAPER_TRUST_THRESHOLD: float = 0.70
PAPER_REVIEW_THRESHOLD: float = 0.40


# ==============================================================================
# RESEARCH WEIGHT PRESETS (From Paper Table III & Notebook Ablations)
# ==============================================================================
RESEARCH_PRESETS: Dict[str, Dict[str, object]] = {
    "Behaviour Dominant (Paper Eq. 6 / Table III Best)": {
        "weights": (0.20, 0.60, 0.20),
        "description": "Selected research candidate. Reaches 0.9531 Balanced Accuracy and 0.9873 F1 in Table III.",
        "note": "Achieves a 16.39 percentage-point recovery over equal fusion and +1.50 points over behaviour-only."
    },
    "Equal Weights (Paper Table III Baseline)": {
        "weights": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "description": "Standard uninformative prior (0.333, 0.333, 0.333). Achieves 0.7892 BA and 0.7628 F1.",
        "note": "Demonstrates signal dilution: combining weaker sources equally significantly drops performance."
    },
    "Reliability-Derived (Paper Eq. 14-15 / Table III)": {
        "weights": (0.3193, 0.3245, 0.3562),
        "description": "Derived from ECE reliability (ri = 1/(1 + λ*ECE), λ=10). Yields 0.7391 BA and 0.7329 F1.",
        "note": "Gives highest nominal weight to network evidence due to low calibration error, hurting downstream classification."
    },
    "ECE-Derived (Paper Table III)": {
        "weights": (0.0693, 0.1316, 0.7992),
        "description": "Strict inverse ECE weighting (assigns 79.9% to network evidence). Achieves 0.4845 BA and 0.4637 F1.",
        "note": "Demonstrates why probability calibration must not be equated with downstream decision utility."
    },
    "Brier-Derived (Paper Table III)": {
        "weights": (0.2501, 0.2908, 0.4591),
        "description": "Inverse Brier score weighting. Achieves 0.7630 BA and 0.7441 F1.",
        "note": "Evaluated in Table III."
    },
    "F1-Derived (Paper Table III)": {
        "weights": (0.3304, 0.3366, 0.3329),
        "description": "Normalized single-source F1 scores across individual baseline models (0.7889 BA).",
        "note": "Close to equal weighting due to high standalone F1 on cleaned single-source datasets."
    },
    "Principled Source-Quality (Ablation 11)": {
        "weights": (0.315067, 0.451161, 0.233771),
        "description": "Balanced compromise combining source diagnostic quality, calibration, and behavioral relevance.",
        "note": "Achieves 0.7988 balanced accuracy without over-relying exclusively on one source."
    }
}


@dataclass
class FusionConfig:
    """Configurable settings for multi-source cybersecurity evidence fusion.

    Attributes:
        static_weight: Base weight assigned to static evidence (DREBIN).
        behaviour_weight: Base weight assigned to behavioural evidence (CICMalDroid).
        network_weight: Base weight assigned to network evidence (CIC-AndMal).
        trusted_threshold: Risk score below which software is classified as 'Trusted'.
        high_risk_threshold: Risk score at or above which software is classified as 'High Risk'.
        trust_trusted_threshold: Trust score at or above which software is 'Trusted' (Paper Eq. 10).
        trust_review_threshold: Trust score below which software is 'High Risk' (Paper Eq. 12).
        decision_basis: Primary decision scheme ('trust' matching Paper Eq. 10-12, or 'risk').
        scale_trust_by_coverage: Whether to scale the trust score by sqrt(coverage).
    """

    static_weight: float = STATIC_WEIGHT
    behaviour_weight: float = BEHAVIOUR_WEIGHT
    network_weight: float = NETWORK_WEIGHT
    trusted_threshold: float = TRUSTED_THRESHOLD
    high_risk_threshold: float = HIGH_RISK_THRESHOLD
    trust_trusted_threshold: float = PAPER_TRUST_THRESHOLD
    trust_review_threshold: float = PAPER_REVIEW_THRESHOLD
    decision_basis: str = "risk"
    scale_trust_by_coverage: bool = True

    def __post_init__(self) -> None:
        """Validate configuration immediately upon creation."""
        self.validate()

    @property
    def weights_tuple(self) -> Tuple[float, float, float]:
        """Return the tuple of (static, behaviour, network) base weights."""
        return (self.static_weight, self.behaviour_weight, self.network_weight)

    def validate(self) -> None:
        """Verify that weights and thresholds conform to physical and mathematical limits.

        Raises:
            ValueError: If weights are negative, don't sum to 1.0, or thresholds are invalid.
        """
        # Validate weights non-negative
        for name, val in [
            ("Static weight", self.static_weight),
            ("Behaviour weight", self.behaviour_weight),
            ("Network weight", self.network_weight),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val):
                raise ValueError(f"{name} must be a real number, got {val}")
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative (>= 0), got {val}")

        # Validate weight sum == 1.0 within numerical tolerance
        w_sum = self.static_weight + self.behaviour_weight + self.network_weight
        if not math.isclose(w_sum, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"Evidence weights must sum to 1.0 (current sum: {w_sum:.6f}). "
                f"Static={self.static_weight:.4f}, Behaviour={self.behaviour_weight:.4f}, Network={self.network_weight:.4f}"
            )

        # Validate risk thresholds
        for name, val in [
            ("Trusted threshold", self.trusted_threshold),
            ("High-risk threshold", self.high_risk_threshold),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val):
                raise ValueError(f"{name} must be a real number, got {val}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")

        if self.trusted_threshold >= self.high_risk_threshold:
            raise ValueError(
                f"Trusted threshold ({self.trusted_threshold:.4f}) must be strictly less than "
                f"High-risk threshold ({self.high_risk_threshold:.4f})."
            )

        # Validate trust thresholds
        for name, val in [
            ("Trust trusted threshold", self.trust_trusted_threshold),
            ("Trust review threshold", self.trust_review_threshold),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val):
                raise ValueError(f"{name} must be a real number, got {val}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")

        if self.trust_review_threshold >= self.trust_trusted_threshold:
            raise ValueError(
                f"Trust review threshold ({self.trust_review_threshold:.4f}) must be strictly less than "
                f"Trust trusted threshold ({self.trust_trusted_threshold:.4f})."
            )

    @classmethod
    def from_preset(cls, preset_name: str, trusted_threshold: float = TRUSTED_THRESHOLD,
                    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
                    trust_trusted_threshold: float = PAPER_TRUST_THRESHOLD,
                    trust_review_threshold: float = PAPER_REVIEW_THRESHOLD) -> "FusionConfig":
        """Factory method to instantiate configuration from a named research preset."""
        if preset_name not in RESEARCH_PRESETS:
            raise KeyError(
                f"Unknown preset: '{preset_name}'. Available: {list(RESEARCH_PRESETS.keys())}"
            )
        w_static, w_beh, w_net = RESEARCH_PRESETS[preset_name]["weights"]  # type: ignore
        return cls(
            static_weight=float(w_static),
            behaviour_weight=float(w_beh),
            network_weight=float(w_net),
            trusted_threshold=trusted_threshold,
            high_risk_threshold=high_risk_threshold,
            trust_trusted_threshold=trust_trusted_threshold,
            trust_review_threshold=trust_review_threshold,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert configuration to dictionary."""
        return {
            "static_weight": self.static_weight,
            "behaviour_weight": self.behaviour_weight,
            "network_weight": self.network_weight,
            "trusted_threshold": self.trusted_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "trust_trusted_threshold": self.trust_trusted_threshold,
            "trust_review_threshold": self.trust_review_threshold,
            "scale_trust_by_coverage": float(self.scale_trust_by_coverage),
        }