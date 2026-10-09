"""Evidence fusion and trust engine for Android Software Trust.

Implements the mathematical formulation defined in:
"Source-Aware Evidence Fusion for Multi-Source Security Risk Assessment"
(Kaur, Goel, Sachdeva, Kumar, Chaudhary, Singhal, 2026).

Equations implemented:
- Eq. (1): R_s, R_b, R_n in [0, 1] (Static, Behavioural, Network risk estimates)
- Eq. (2) & (3): Fused risk R_f = w_s*R_s + w_b*R_b + w_n*R_n with sum(w_i) = 1, w_i >= 0
- Eq. (4): Available weight renormalization \tilde{w}_i = (w_i * I_i) / sum(w_j * I_j)
- Eq. (5): Fused risk with incomplete evidence R_f = sum_{i in A} \tilde{w}_i * R_i
- Eq. (7): Weighted disagreement D = sum_{i in A} \tilde{w}_i * (R_i - R_f)^2
- Eq. (8): Evidence coverage C = sum_{i in A} w_i
- Eq. (9): Trust-oriented decision score T = (1 - R_f) * (1 - D) * sqrt(C)
- Eq. (10, 11, 12): Trust decision operating states:
    T >= 0.70 => Trusted
    0.40 <= T < 0.70 => Review
    T < 0.40 => High Risk
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

from src.config import (
    FusionConfig,
    STATIC_WEIGHT,
    BEHAVIOUR_WEIGHT,
    NETWORK_WEIGHT,
    TRUSTED_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    PAPER_TRUST_THRESHOLD,
    PAPER_REVIEW_THRESHOLD,
)


@dataclass
class SourceContribution:
    """Detailed contribution of a single evidence source."""
    source_name: str
    probability: float
    base_weight: float
    effective_weight: float
    risk_contribution: float
    deviation_from_mean: float
    disagreement_contribution: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "source": self.source_name,
            "probability": self.probability,
            "base_weight": self.base_weight,
            "effective_weight": self.effective_weight,
            "risk_contribution": self.risk_contribution,
            "deviation_from_mean": self.deviation_from_mean,
            "disagreement_contribution": self.disagreement_contribution,
        }


@dataclass
class FusionResult:
    """Structured assessment result produced by the evidence fusion engine.

    Attributes:
        risk: Fused suspiciousness risk R_f in [0, 1] (Eq. 2 & 5). Higher indicates more suspicious.
        disagreement: Weighted variance across available sources D in [0, 0.25] (Eq. 7).
        coverage: Proportion of base evidence weight available C in [0, 1] (Eq. 8).
        trust: Benign-oriented decision-support score T in [0, 1] (Eq. 9). Higher indicates safer.
        decision: Primary decision state ('Trusted', 'Review', 'High Risk', 'No Evidence').
        trust_decision: Decision state derived from Paper Trust thresholds (Eq. 10-12).
        risk_decision: Decision state derived from Risk thresholds (R_f < 0.20, R_f >= 0.50).
        decision_reason: Plain-language explanation of the decision rationale.
        has_evidence: Whether at least one valid evidence score was provided.
        available_sources: List of source names that contributed to the assessment.
        missing_sources: List of source names that were omitted or missing.
        contributions: Detailed per-source contributions.
        warnings: Contextual alerts (e.g., high disagreement, degraded coverage).
        config_snapshot: The configuration used for this calculation.
    """
    risk: Optional[float]
    disagreement: Optional[float]
    coverage: float
    trust: Optional[float]
    decision: str
    decision_reason: str
    has_evidence: bool
    trust_decision: str = "No Evidence"
    risk_decision: str = "No Evidence"
    available_sources: List[str] = field(default_factory=list)
    missing_sources: List[str] = field(default_factory=list)
    contributions: Dict[str, SourceContribution] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    config_snapshot: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "risk": round(self.risk, 4) if self.risk is not None else None,
            "disagreement": round(self.disagreement, 4) if self.disagreement is not None else None,
            "coverage": round(self.coverage, 4),
            "trust": round(self.trust, 4) if self.trust is not None else None,
            "decision": self.decision,
            "trust_decision": self.trust_decision,
            "risk_decision": self.risk_decision,
            "decision_reason": self.decision_reason,
            "has_evidence": self.has_evidence,
            "available_sources": self.available_sources,
            "missing_sources": self.missing_sources,
            "warnings": self.warnings,
            "config": self.config_snapshot,
        }


def validate_score(score: Optional[float], source_name: str) -> Optional[float]:
    """Validate that a score is either None, NaN, or a float in [0.0, 1.0].

    Raises:
        ValueError: If score is out of range or of invalid type.
    """
    if score is None:
        return None
    if isinstance(score, (int, float)) and math.isnan(score):
        return None
    if not isinstance(score, (int, float)):
        raise ValueError(
            f"{source_name} score must be a numeric probability, got {type(score).__name__}: {score}"
        )
    val = float(score)
    if not (0.0 <= val <= 1.0):
        raise ValueError(
            f"{source_name} score must be in range [0.0, 1.0], got {val}"
        )
    return val


def fuse_evidence(
    static_prob: Optional[float] = None,
    behaviour_prob: Optional[float] = None,
    network_prob: Optional[float] = None,
    config: Optional[FusionConfig] = None,
) -> FusionResult:
    """Fuse available evidence sources using the paper's source-aware framework.

    Args:
        static_prob: Static evidence risk R_s (DREBIN).
        behaviour_prob: Behavioural evidence risk R_b (CICMalDroid 2020).
        network_prob: Network evidence risk R_n (CIC-AndMal2017).
        config: Fusion configuration. If None, default FusionConfig() is used.

    Returns:
        FusionResult object with risk, disagreement, coverage, trust, and decision states.
    """
    if config is None:
        config = FusionConfig()
    else:
        config.validate()

    # Validate inputs
    s_val = validate_score(static_prob, "Static")
    b_val = validate_score(behaviour_prob, "Behavioural")
    n_val = validate_score(network_prob, "Network")

    raw_inputs = [
        ("Static", s_val, config.static_weight),
        ("Behavioural", b_val, config.behaviour_weight),
        ("Network", n_val, config.network_weight),
    ]

    available_items = [(name, val, w) for name, val, w in raw_inputs if val is not None]
    missing_sources = [name for name, val, _ in raw_inputs if val is None]
    available_sources = [name for name, _, _ in available_items]

    # Handle no evidence
    if not available_items:
        return FusionResult(
            risk=None,
            disagreement=None,
            coverage=0.0,
            trust=None,
            decision="No Evidence",
            trust_decision="No Evidence",
            risk_decision="No Evidence",
            decision_reason="No evidence scores were provided. Assessment cannot be computed.",
            has_evidence=False,
            available_sources=[],
            missing_sources=missing_sources,
            contributions={},
            warnings=["No evidence available. Please input at least one source probability."],
            config_snapshot=config.to_dict(),
        )

    # Eq. (8): Evidence coverage C = sum_{i in A} w_i
    coverage = sum(w for _, _, w in available_items)

    # Eq. (4): Available weight renormalization \tilde{w}_i = (w_i * I_i) / sum(w_j * I_j)
    total_avail_weight = sum(w for _, _, w in available_items)
    if total_avail_weight <= 0.0:
        raise ValueError("Sum of available evidence weights is non-positive.")

    norm_items = [
        (name, val, base_w, base_w / total_avail_weight)
        for name, val, base_w in available_items
    ]

    # Eq. (5): Fused suspiciousness risk R_f = sum_{i in A} \tilde{w}_i * R_i
    fused_risk = sum(eff_w * val for _, val, _, eff_w in norm_items)
    fused_risk = max(0.0, min(1.0, float(fused_risk)))

    # Eq. (7): Weighted disagreement D = sum_{i in A} \tilde{w}_i * (R_i - R_f)^2
    if len(available_items) > 1:
        disagreement = sum(
            eff_w * ((val - fused_risk) ** 2)
            for _, val, _, eff_w in norm_items
        )
        disagreement = max(0.0, min(0.25, float(disagreement)))
    else:
        disagreement = 0.0

    # Eq. (9): Trust-oriented decision score T = (1 - R_f) * (1 - D) * sqrt(C)
    base_trust = 1.0 - fused_risk
    consistency_penalty = 1.0 - disagreement
    cov_factor = math.sqrt(coverage) if config.scale_trust_by_coverage else 1.0
    trust = max(0.0, min(1.0, float(base_trust * consistency_penalty * cov_factor)))

    # --------------------------------------------------------------------------
    # Decision State A: Paper Trust Decision (Eq. 10, 11, 12)
    # --------------------------------------------------------------------------
    if trust >= config.trust_trusted_threshold:
        trust_decision = "Trusted"
    elif trust >= config.trust_review_threshold:
        trust_decision = "Review"
    else:
        trust_decision = "High Risk"

    # --------------------------------------------------------------------------
    # Decision State B: Config Risk Decision (Thresholds on R_f)
    # --------------------------------------------------------------------------
    if fused_risk < config.trusted_threshold:
        risk_decision = "Trusted"
    elif fused_risk >= config.high_risk_threshold:
        risk_decision = "High Risk"
    else:
        risk_decision = "Review"

    # Default decision follows config.decision_basis
    if getattr(config, "decision_basis", "risk") == "trust":
        decision = trust_decision
        decision_reason = (
            f"Paper Trust Score T={trust:.3f} (Eq. 9). "
            f"Evaluated against Paper Eq. (10-12) thresholds [T_trusted={config.trust_trusted_threshold:.2f}, T_review={config.trust_review_threshold:.2f}] => {trust_decision}."
        )
    else:
        decision = risk_decision
        if fused_risk < config.trusted_threshold:
            decision_reason = (
                f"Fused risk R_f={fused_risk:.3f} is below the trusted threshold ({config.trusted_threshold:.2f}). "
                f"Evidence indicates predominantly benign characteristics."
            )
        elif fused_risk >= config.high_risk_threshold:
            decision_reason = (
                f"Fused risk R_f={fused_risk:.3f} meets or exceeds the high-risk threshold ({config.high_risk_threshold:.2f}). "
                f"Evidence strongly suggests malicious capabilities."
            )
        else:
            decision_reason = (
                f"Fused risk R_f={fused_risk:.3f} lies in the ambiguous review interval "
                f"[{config.trusted_threshold:.2f}, {config.high_risk_threshold:.2f}). "
                f"Manual inspection or sandboxed execution is recommended."
            )

    # Contextual Warnings (Paper Section VI-C)
    warnings: List[str] = []
    if disagreement >= 0.10:
        warnings.append(
            f"High source disagreement detected (D={disagreement:.4f}). "
            f"Conflicting source signals detected. Trust score is penalized from {base_trust:.3f} to {trust:.3f}."
        )
        if risk_decision == "Trusted" and trust_decision != "Trusted":
            warnings.append(
                f"Section VI-C Mechanism Alert: While Fused Risk is low ({fused_risk:.3f}), "
                f"high source disagreement (D={disagreement:.4f}) correctly prevents a 'Trusted' classification under Paper Eq. (11)!"
            )

    if coverage < 1.0:
        warnings.append(
            f"Incomplete evidence coverage (C={coverage * 100:.1f}%). "
            f"Missing sources: {', '.join(missing_sources)}. Weights renormalized via Eq. (4)."
        )

    # Detailed Source Contributions
    contributions: Dict[str, SourceContribution] = {}
    for name, val, base_w, eff_w in norm_items:
        dev = (val - fused_risk) ** 2
        risk_c = eff_w * val
        disagree_c = eff_w * dev if len(available_items) > 1 else 0.0
        contributions[name] = SourceContribution(
            source_name=name,
            probability=val,
            base_weight=base_w,
            effective_weight=eff_w,
            risk_contribution=risk_c,
            deviation_from_mean=dev,
            disagreement_contribution=disagree_c,
        )

    return FusionResult(
        risk=fused_risk,
        disagreement=disagreement,
        coverage=coverage,
        trust=trust,
        decision=decision,
        trust_decision=trust_decision,
        risk_decision=risk_decision,
        decision_reason=decision_reason,
        has_evidence=True,
        available_sources=available_sources,
        missing_sources=missing_sources,
        contributions=contributions,
        warnings=warnings,
        config_snapshot=config.to_dict(),
    )
