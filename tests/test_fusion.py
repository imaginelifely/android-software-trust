"""Comprehensive unit tests for evidence fusion engine and configuration."""

import pytest
import math
import pandas as pd

from src.config import (
    FusionConfig,
    RESEARCH_PRESETS,
    STATIC_WEIGHT,
    BEHAVIOUR_WEIGHT,
    NETWORK_WEIGHT,
    TRUSTED_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)
from src.fusion import (
    fuse_evidence,
    validate_score,
    FusionResult,
    SourceContribution,
)
from src.history import AssessmentHistory, AssessmentRecord


# ==============================================================================
# 1. CONFIGURATION TESTS
# ==============================================================================
class TestConfiguration:
    def test_default_config_values(self):
        cfg = FusionConfig()
        assert cfg.static_weight == 0.20
        assert cfg.behaviour_weight == 0.60
        assert cfg.network_weight == 0.20
        assert cfg.trusted_threshold == 0.20
        assert cfg.high_risk_threshold == 0.50
        assert cfg.scale_trust_by_coverage is True

    def test_weight_sum_validation(self):
        # Weights not summing to 1.0 should raise ValueError
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FusionConfig(static_weight=0.3, behaviour_weight=0.3, network_weight=0.3)

    def test_negative_weight_validation(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            FusionConfig(static_weight=-0.1, behaviour_weight=0.8, network_weight=0.3)

    def test_threshold_ordering_validation(self):
        # trusted >= high_risk should raise ValueError
        with pytest.raises(ValueError, match="strictly less than"):
            FusionConfig(trusted_threshold=0.50, high_risk_threshold=0.50)

        with pytest.raises(ValueError, match="strictly less than"):
            FusionConfig(trusted_threshold=0.60, high_risk_threshold=0.40)

    def test_threshold_out_of_bounds(self):
        with pytest.raises(ValueError, match="in range"):
            FusionConfig(trusted_threshold=-0.05, high_risk_threshold=0.50)

        with pytest.raises(ValueError, match="in range"):
            FusionConfig(trusted_threshold=0.20, high_risk_threshold=1.05)

    def test_research_presets(self):
        for name in RESEARCH_PRESETS:
            cfg = FusionConfig.from_preset(name)
            w_sum = cfg.static_weight + cfg.behaviour_weight + cfg.network_weight
            assert math.isclose(w_sum, 1.0, abs_tol=1e-4)


# ==============================================================================
# 2. EVIDENCE FUSION TESTS
# ==============================================================================
class TestFusionEngine:
    def test_all_three_sources_present(self):
        cfg = FusionConfig(static_weight=0.2, behaviour_weight=0.6, network_weight=0.2)
        # 0.2*0.1 + 0.6*0.3 + 0.2*0.2 = 0.02 + 0.18 + 0.04 = 0.24
        res = fuse_evidence(0.10, 0.30, 0.20, config=cfg)
        assert res.has_evidence is True
        assert res.coverage == 1.0
        assert math.isclose(res.risk, 0.24, abs_tol=1e-4)
        assert res.decision == "Review"
        assert len(res.available_sources) == 3
        assert len(res.missing_sources) == 0

    def test_one_source_missing(self):
        cfg = FusionConfig(static_weight=0.2, behaviour_weight=0.6, network_weight=0.2)
        # Network missing: available weights are 0.2 and 0.6 (sum 0.8)
        # Renormalized: static = 0.2/0.8 = 0.25, behaviour = 0.6/0.8 = 0.75
        # risk = 0.25*0.20 + 0.75*0.80 = 0.05 + 0.60 = 0.65
        res = fuse_evidence(static_prob=0.20, behaviour_prob=0.80, network_prob=None, config=cfg)
        assert res.has_evidence is True
        assert math.isclose(res.coverage, 0.80, abs_tol=1e-4)
        assert math.isclose(res.risk, 0.65, abs_tol=1e-4)
        assert res.decision == "High Risk"
        assert "Network" in res.missing_sources
        assert "Static" in res.available_sources
        assert "Behavioural" in res.available_sources

    def test_two_sources_missing(self):
        cfg = FusionConfig(static_weight=0.2, behaviour_weight=0.6, network_weight=0.2)
        # Only static present
        res = fuse_evidence(static_prob=0.15, behaviour_prob=None, network_prob=None, config=cfg)
        assert res.has_evidence is True
        assert math.isclose(res.coverage, 0.20, abs_tol=1e-4)
        assert math.isclose(res.risk, 0.15, abs_tol=1e-4)
        assert res.disagreement == 0.0  # Single source has zero disagreement
        assert res.decision == "Trusted"
        assert len(res.available_sources) == 1

    def test_no_evidence_available(self):
        cfg = FusionConfig()
        res = fuse_evidence(None, None, None, config=cfg)
        assert res.has_evidence is False
        assert res.risk is None
        assert res.disagreement is None
        assert res.trust is None
        assert res.coverage == 0.0
        assert res.decision == "No Evidence"
        assert len(res.available_sources) == 0
        assert len(res.missing_sources) == 3
        assert len(res.warnings) > 0

    def test_invalid_probability_inputs(self):
        cfg = FusionConfig()
        with pytest.raises(ValueError, match="in range"):
            fuse_evidence(1.5, 0.5, 0.2, config=cfg)

        with pytest.raises(ValueError, match="in range"):
            fuse_evidence(-0.1, 0.5, 0.2, config=cfg)

        with pytest.raises(ValueError, match="must be a numeric probability"):
            fuse_evidence("malicious", 0.5, 0.2, config=cfg)  # type: ignore

    def test_high_disagreement_scenario(self):
        cfg = FusionConfig(static_weight=0.5, behaviour_weight=0.5, network_weight=0.0)
        # Complete conflict: 0.0 vs 1.0 with equal weights -> risk = 0.5, disagreement = 0.5*(0.25) + 0.5*(0.25) = 0.25
        res = fuse_evidence(0.0, 1.0, None, config=cfg)
        assert math.isclose(res.risk, 0.50, abs_tol=1e-4)
        assert math.isclose(res.disagreement, 0.25, abs_tol=1e-4)
        assert any("High source disagreement" in w for w in res.warnings)

    def test_trusted_decision(self):
        cfg = FusionConfig()
        # All sources indicate very low suspiciousness (< 0.20)
        res = fuse_evidence(0.05, 0.10, 0.08, config=cfg)
        assert res.risk < cfg.trusted_threshold
        assert res.decision == "Trusted"
        assert res.trust > 0.80

    def test_review_decision(self):
        cfg = FusionConfig()
        # Risk between 0.20 and 0.50
        res = fuse_evidence(0.30, 0.35, 0.32, config=cfg)
        assert cfg.trusted_threshold <= res.risk < cfg.high_risk_threshold
        assert res.decision == "Review"

    def test_high_risk_decision(self):
        cfg = FusionConfig()
        # Risk >= 0.50
        res = fuse_evidence(0.80, 0.90, 0.85, config=cfg)
        assert res.risk >= cfg.high_risk_threshold
        assert res.decision == "High Risk"
        assert res.trust < 0.20

    def test_boundary_probabilities(self):
        cfg = FusionConfig()
        # All 0.0
        res_zero = fuse_evidence(0.0, 0.0, 0.0, config=cfg)
        assert res_zero.risk == 0.0
        assert res_zero.disagreement == 0.0
        assert res_zero.trust == 1.0
        assert res_zero.decision == "Trusted"

        # All 1.0
        res_one = fuse_evidence(1.0, 1.0, 1.0, config=cfg)
        assert res_one.risk == 1.0
        assert res_one.disagreement == 0.0
        assert res_one.trust == 0.0
        assert res_one.decision == "High Risk"


# ==============================================================================
# 3. ASSESSMENT HISTORY TESTS
# ==============================================================================
class TestAssessmentHistory:
    def test_history_logging_and_export(self):
        hist = AssessmentHistory()
        assert hist.count() == 0

        res1 = fuse_evidence(0.1, 0.2, 0.15)
        rec1 = hist.add_record(res1, 0.1, 0.2, 0.15, label="Test Sample A")

        assert hist.count() == 1
        assert rec1.decision == "Trusted"

        res2 = fuse_evidence(0.9, 0.8, 0.95)
        hist.add_record(res2, 0.9, 0.8, 0.95, label="Test Sample B")
        assert hist.count() == 2

        # Check DataFrame
        df = hist.to_dataframe()
        assert len(df) == 2
        assert list(df["decision"]) == ["Trusted", "High Risk"]

        # Check CSV export
        csv_str = hist.to_csv()
        assert "Test Sample A" in csv_str
        assert "Test Sample B" in csv_str
        assert "fused_risk" in csv_str

        # Clear history
        hist.clear()
        assert hist.count() == 0
        empty_df = hist.to_dataframe()
        assert len(empty_df) == 0


# ==============================================================================
# 4. ADDITIONAL EDGE CASES & NUMERICAL BOUNDARIES
# ==============================================================================
class TestAdditionalEdgeCases:
    def test_extreme_disagreement_bounds(self):
        # 0.0 and 1.0 with equal weights yields maximum variance: 0.25
        cfg = FusionConfig(static_weight=0.5, behaviour_weight=0.5, network_weight=0.0)
        res = fuse_evidence(0.0, 1.0, None, config=cfg)
        assert res.disagreement == 0.25
        assert res.risk == 0.50
        assert res.trust == (1.0 - 0.50) * (1.0 - 0.25) * 1.0
        assert math.isclose(res.trust, 0.375, abs_tol=1e-4)

    def test_source_contributions_breakdown(self):
        cfg = FusionConfig(static_weight=0.2, behaviour_weight=0.6, network_weight=0.2)
        res = fuse_evidence(0.2, 0.8, 0.4, config=cfg)
        assert len(res.contributions) == 3
        
        c_beh = res.contributions["Behavioural"]
        assert c_beh.probability == 0.8
        assert c_beh.effective_weight == 0.6
        assert math.isclose(c_beh.risk_contribution, 0.48, abs_tol=1e-4)

    def test_float_tolerance_in_weights(self):
        # 0.3333333333333333 each is fine
        cfg = FusionConfig(static_weight=1/3, behaviour_weight=1/3, network_weight=1/3)
        assert cfg.static_weight + cfg.behaviour_weight + cfg.network_weight == 1.0

    def test_scale_trust_disabled(self):
        cfg = FusionConfig(static_weight=0.5, behaviour_weight=0.5, network_weight=0.0, scale_trust_by_coverage=False)
        # Only static present -> coverage = 0.5
        res = fuse_evidence(0.2, None, None, config=cfg)
        # Without coverage scaling: trust = (1 - 0.2) * (1 - 0.0) = 0.8
        assert math.isclose(res.trust, 0.80, abs_tol=1e-4)

    def test_scale_trust_enabled(self):
        cfg = FusionConfig(static_weight=0.5, behaviour_weight=0.5, network_weight=0.0, scale_trust_by_coverage=True)
        # Only static present -> coverage = 0.5
        res = fuse_evidence(0.2, None, None, config=cfg)
        # With coverage scaling: trust = 0.80 * sqrt(0.5) = 0.565685
        expected = 0.80 * math.sqrt(0.5)
        assert math.isclose(res.trust, expected, abs_tol=1e-4)


# ==============================================================================
# 5. PAPER EQUATIONS & DUAL DECISION TESTS (Equations 1-12)
# ==============================================================================
class TestPaperEquations:
    def test_paper_trust_thresholds(self):
        # T >= 0.70 -> Trusted (Eq. 10)
        # 0.40 <= T < 0.70 -> Review (Eq. 11)
        # T < 0.40 -> High Risk (Eq. 12)
        cfg_trust = FusionConfig(decision_basis="trust")
        
        # High confidence safe: risk = 0.05, D = 0.0, Cov = 1.0 -> T = 0.95 >= 0.70 => Trusted
        res_safe = fuse_evidence(0.05, 0.05, 0.05, config=cfg_trust)
        assert res_safe.trust_decision == "Trusted"
        assert res_safe.decision == "Trusted"

        # Ambiguous: risk = 0.45, D = 0.0, Cov = 1.0 -> T = 0.55 in [0.40, 0.70) => Review
        res_mid = fuse_evidence(0.45, 0.45, 0.45, config=cfg_trust)
        assert res_mid.trust_decision == "Review"
        assert res_mid.decision == "Review"

        # Malicious: risk = 0.85, D = 0.0, Cov = 1.0 -> T = 0.15 < 0.40 => High Risk
        res_mal = fuse_evidence(0.85, 0.85, 0.85, config=cfg_trust)
        assert res_mal.trust_decision == "High Risk"
        assert res_mal.decision == "High Risk"

    def test_section_6c_disagreement_validation(self):
        # Section VI-C: low risk + high disagreement routed to Review, never Trusted
        cfg_trust = FusionConfig(static_weight=0.5, behaviour_weight=0.5, network_weight=0.0, decision_basis="trust")
        # Conflict: 0.10 and 0.60 -> Fused risk = 0.35 (fairly low risk), but D = 0.0625
        res_conflict = fuse_evidence(0.10, 0.60, None, config=cfg_trust)
        assert res_conflict.risk < 0.50
        assert res_conflict.disagreement > 0.05
        # Under trust, it drops into Review:
        assert res_conflict.trust < 0.70
        assert res_conflict.trust_decision == "Review"
