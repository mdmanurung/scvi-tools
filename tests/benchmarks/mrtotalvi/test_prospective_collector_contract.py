"""Pure contracts for prospective collector continuation semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from benchmarks.mrtotalvi.comparator import (
    _add_prospective_representation_metrics,
    _mrtotalvi_residual_gradient_diagnostics,
    _prediction_metrics_with_integrity_continuation,
    _representation_payload,
    _validation_arrays_with_integrity_continuation,
)


def test_prospective_export_preserves_nonfinite_evidence_only_when_requested():
    cells = np.asarray(["c0", "c1", "c2"])
    values = np.asarray([[0.0], [np.nan], [1.0]])

    with pytest.raises(ValueError, match="must be finite"):
        _representation_payload(cells, values)

    preserved = _representation_payload(
        cells,
        values,
        allow_integrity_failures=True,
    )
    assert np.isnan(preserved["values"][1, 0])


def test_missing_registered_residual_structure_is_serialized_prospectively():
    model = SimpleNamespace(module=SimpleNamespace())

    assert _mrtotalvi_residual_gradient_diagnostics(
        model,
        indices=np.asarray([0, 1, 2]),
        batch_size=3,
        allow_integrity_failures=True,
    ) == (None, None)

    with pytest.raises(ValueError, match="no registered residual"):
        _mrtotalvi_residual_gradient_diagnostics(
            model,
            indices=np.asarray([0, 1, 2]),
            batch_size=3,
        )


def test_invalid_scale_keeps_computable_embedding_summaries(
    monkeypatch,
):
    metrics = {
        "factual_z_posterior_scale": None,
        "factual_z_latent_variance": 0.2,
        "factual_z_effective_rank": 2.0,
    }
    representation = {
        "factual_z": {
            "cell_ids": np.asarray(["c0", "c1", "c2"]),
            "values": np.asarray([[0.0], [1.0], [2.0]]),
        }
    }
    latent = {
        "factual_z": {
            "representation_all_finite": 1.0,
            "exact_nonconstant_variation": 1.0,
            "posterior_scales_all_valid": 0.0,
        }
    }

    def annotation_metrics(payload, *_args, **_kwargs):
        updated = dict(payload)
        updated["factual_z_state_balanced_accuracy"] = 0.5
        updated["factual_z_knn_state_accuracy_k15"] = 0.5
        updated["factual_z_technical_batch_mixing"] = 0.5
        return updated

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator.add_evaluation_annotation_metrics",
        annotation_metrics,
    )

    updated, no_calls = _add_prospective_representation_metrics(
        metrics,
        representation,
        latent_by_representation=latent,
        gradient_coverage=1.0,
        state_labels=np.asarray(["a", "a", "b"]),
        sample_labels=np.asarray(["s0", "s1", "s0"]),
        technical_batch_labels=np.asarray(["b0", "b0", "b1"]),
        random_state=7,
    )

    assert updated["factual_z_state_balanced_accuracy"] == 0.5
    assert no_calls == {
        "factual_z_posterior_scale": [
            "factual_z_posterior_scales_all_valid"
        ]
    }


@pytest.mark.parametrize(
    ("candidate_id", "representations", "reason"),
    [
        (
            "B1",
            {
                "factual_z": {
                    "values": np.asarray([[0.0], [np.nan], [1.0]])
                }
            },
            "factual_z_representation_all_finite",
        ),
        (
            "D0",
            {
                "u": {"values": np.asarray([[0.0], [np.nan], [1.0]])},
                "factual_z": {
                    "values": np.asarray([[0.0], [1.0], [2.0]])
                },
            },
            "u_representation_all_finite",
        ),
    ],
)
def test_nonfinite_stock_and_mrtotalvi_exports_continue_prediction_no_calls(
    candidate_id,
    representations,
    reason,
):
    diagnostic = Mock(side_effect=ValueError("predictive values must be finite"))

    prediction, no_calls = _prediction_metrics_with_integrity_continuation(
        diagnostic,
        candidate_id=candidate_id,
        representations=representations,
    )

    diagnostic.assert_called_once_with()
    assert set(prediction) == {
        "rna_negative_log_likelihood",
        "protein_negative_log_likelihood",
        "multimodal_predictive_loss",
        "rna_calibration_error",
        "protein_calibration_error",
    }
    assert all(value is None for value in prediction.values())
    assert no_calls == {
        metric_id: [reason]
        for metric_id in (
            "rna_reconstruction_loss",
            "rna_heldout_negative_log_likelihood",
            "rna_posterior_predictive_calibration",
            "protein_reconstruction_loss",
            "protein_heldout_negative_log_likelihood",
            "multimodal_heldout_predictive_loss",
            "protein_posterior_predictive_calibration",
        )
    }


@pytest.mark.parametrize(
    ("candidate_id", "representations", "expected_no_call_ids"),
    [
        (
            "B1",
            {"factual_z": {"values": np.asarray([[0.0], [np.nan], [1.0]])}},
            {
                "kl_z",
                "rna_reconstruction_loss",
                "rna_heldout_negative_log_likelihood",
                "rna_posterior_predictive_calibration",
                "protein_reconstruction_loss",
                "protein_heldout_negative_log_likelihood",
                "multimodal_heldout_predictive_loss",
                "protein_posterior_predictive_calibration",
            },
        ),
        (
            "D0",
            {
                "u": {"values": np.asarray([[0.0], [np.nan], [1.0]])},
                "factual_z": {"values": np.asarray([[0.0], [1.0], [2.0]])},
            },
            {
                "kl_z",
                "kl_u",
                "registered_residual_magnitude",
                "centering_max_abs",
                "rna_reconstruction_loss",
                "rna_heldout_negative_log_likelihood",
                "rna_posterior_predictive_calibration",
                "protein_reconstruction_loss",
                "protein_heldout_negative_log_likelihood",
                "multimodal_heldout_predictive_loss",
                "protein_posterior_predictive_calibration",
            },
        ),
    ],
)
def test_nonfinite_stock_and_mrtotalvi_exports_continue_validation_no_calls(
    candidate_id,
    representations,
    expected_no_call_ids,
):
    diagnostic = Mock(side_effect=ValueError("validation arrays are nonfinite"))

    arrays, no_calls = _validation_arrays_with_integrity_continuation(
        diagnostic,
        candidate_id=candidate_id,
        representations=representations,
    )

    diagnostic.assert_called_once_with()
    assert arrays is None
    assert set(no_calls) == expected_no_call_ids
    expected_reason = (
        "factual_z_representation_all_finite"
        if candidate_id == "B1"
        else "u_representation_all_finite"
    )
    assert set(map(tuple, no_calls.values())) == {(expected_reason,)}


def test_prediction_error_without_declared_nonfinite_export_is_not_suppressed():
    diagnostic = Mock(side_effect=ValueError("unrelated validation failure"))

    with pytest.raises(ValueError, match="unrelated validation failure"):
        _prediction_metrics_with_integrity_continuation(
            diagnostic,
            candidate_id="B1",
            representations={
                "factual_z": {
                    "values": np.asarray([[0.0], [1.0], [2.0]])
                }
            },
        )


@pytest.mark.parametrize(
    ("continuation", "error"),
    [
        (
            _prediction_metrics_with_integrity_continuation,
            RuntimeError("predictive backend failed"),
        ),
        (
            _validation_arrays_with_integrity_continuation,
            RuntimeError("predictive backend failed"),
        ),
        (
            _prediction_metrics_with_integrity_continuation,
            ValueError("missing required registry field"),
        ),
        (
            _validation_arrays_with_integrity_continuation,
            ValueError("missing required registry field"),
        ),
    ],
)
def test_structural_error_is_not_suppressed_by_nonfinite_export(
    continuation,
    error,
):
    diagnostic = Mock(side_effect=error)

    with pytest.raises(type(error), match=str(error)):
        continuation(
            diagnostic,
            candidate_id="D0",
            representations={
                "u": {"values": np.asarray([[0.0], [np.nan], [1.0]])},
                "factual_z": {
                    "values": np.asarray([[0.0], [1.0], [2.0]])
                },
            },
        )
