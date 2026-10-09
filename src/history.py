"""Assessment history tracking and export module.

Stores manual risk assessments locally in session/file storage,
allowing analysts to review, audit, export (CSV), and clear assessments.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
import io
import pandas as pd

from src.fusion import FusionResult


@dataclass
class AssessmentRecord:
    """Stored record of a completed risk assessment."""
    id: str
    timestamp: str
    static_prob: Optional[float]
    behaviour_prob: Optional[float]
    network_prob: Optional[float]
    fused_risk: Optional[float]
    disagreement: Optional[float]
    coverage: float
    trust: Optional[float]
    decision: str
    decision_reason: str
    static_weight: float
    behaviour_weight: float
    network_weight: float
    trusted_threshold: float
    high_risk_threshold: float
    warnings: str
    label: Optional[str] = None


class AssessmentHistory:
    """Manager for local risk assessment history."""

    def __init__(self) -> None:
        self.records: List[AssessmentRecord] = []

    def add_record(
        self,
        result: FusionResult,
        static_prob: Optional[float],
        behaviour_prob: Optional[float],
        network_prob: Optional[float],
        label: Optional[str] = None,
    ) -> AssessmentRecord:
        """Add a completed FusionResult to the local history."""
        rec_id = f"AST-{len(self.records) + 1:04d}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cfg = result.config_snapshot or {}
        record = AssessmentRecord(
            id=rec_id,
            timestamp=now_str,
            static_prob=static_prob,
            behaviour_prob=behaviour_prob,
            network_prob=network_prob,
            fused_risk=result.risk,
            disagreement=result.disagreement,
            coverage=result.coverage,
            trust=result.trust,
            decision=result.decision,
            decision_reason=result.decision_reason,
            static_weight=cfg.get("static_weight", 0.20),
            behaviour_weight=cfg.get("behaviour_weight", 0.60),
            network_weight=cfg.get("network_weight", 0.20),
            trusted_threshold=cfg.get("trusted_threshold", 0.20),
            high_risk_threshold=cfg.get("high_risk_threshold", 0.50),
            warnings="; ".join(result.warnings) if result.warnings else "None",
            label=label or f"Assessment #{len(self.records) + 1}",
        )
        self.records.append(record)
        return record

    def clear(self) -> None:
        """Clear all stored assessment history records."""
        self.records.clear()

    def count(self) -> int:
        """Return the number of stored assessments."""
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert stored records into a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "id", "timestamp", "label", "static_prob", "behaviour_prob", "network_prob",
                "fused_risk", "disagreement", "coverage", "trust", "decision",
                "static_weight", "behaviour_weight", "network_weight",
                "trusted_threshold", "high_risk_threshold", "warnings", "decision_reason"
            ])
        return pd.DataFrame([asdict(r) for r in self.records])

    def to_csv(self) -> str:
        """Export history records as a CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)
