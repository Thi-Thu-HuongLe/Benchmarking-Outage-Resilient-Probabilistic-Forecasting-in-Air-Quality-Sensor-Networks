from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_third_development import (
    calibration_factors,
    decision_record,
    reframing_assessment,
)


def test_calibration_factors_loads_locked_condition(tmp_path: Path) -> None:
    values = np.full((2, 3), 1.25, dtype=np.float32)
    np.savez_compressed(
        tmp_path / "calibration.npz",
        single_station_trailing_6h__combined_factors=values,
    )
    np.testing.assert_allclose(calibration_factors(tmp_path, 6), values)


def test_decision_rejects_candidate_that_loses_equal_training_control() -> None:
    protocol = {
        "candidate": "adaptive",
        "candidate_training_control": "equal_training",
        "base_model_for_candidate_and_control": "base",
    }
    primary = pd.DataFrame(
        [
            {
                "comparator": comparator,
                "relative_benefit_percent": benefit,
                "candidate_better_ci95": False,
                "dm_significant_holm_0_05": False,
                "mean_difference_candidate_minus_comparator": -benefit,
            }
            for comparator, benefit in (
                [("equal_training", 0.1)] * 3 + [("equal_training", -0.1)] + [("base", 0.1)] * 4
            )
        ]
    )
    per_seed = pd.DataFrame(
        [
            {"comparator": comparator, "candidate_win": True}
            for comparator in ("equal_training", "base")
            for _ in range(20)
        ]
    )
    guardrails = pd.DataFrame({"guardrail_pass": [True, True]})
    decision = decision_record(
        protocol=protocol,
        primary=primary,
        per_seed=per_seed,
        guardrails=guardrails,
    )
    assert decision["advance_adaptive_graph_to_holdout"] is False
    assert decision["decision"] == "DO_NOT_ADVANCE_ADAPTIVE_GRAPH"


def test_reframing_assessment_keeps_dominated_control_as_control() -> None:
    primary = pd.DataFrame(
        {
            "comparator": ["stronger"] * 4,
            "mean_difference_candidate_minus_comparator": [0.1] * 4,
            "relative_benefit_percent": [-1.0] * 4,
        }
    )
    assessment = reframing_assessment(
        primary,
        assessed_model="equal_training",
        prespecified_candidate="adaptive",
    )
    assert assessment["uniformly_dominated_by"] == ["stronger"]
    assert assessment["holdout_action_authorized"] is False
