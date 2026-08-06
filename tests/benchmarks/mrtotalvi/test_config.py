"""Behavioral contracts for the frozen MrTotalVI benchmark factorial."""

from __future__ import annotations

from benchmarks.mrtotalvi import candidate_configs


def test_candidate_factorial_changes_only_preregistered_axes():
    """C0-C4 expose exactly the frozen, auditable model differences."""
    candidates = candidate_configs()

    assert tuple(candidates) == ("C0", "C1", "C2", "C3", "C4")
    assert candidates["C0"].scientific_role == "accepted baseline"
    assert candidates["C4"].scientific_role == "only primary-DA-eligible v2 candidate"

    c1 = candidates["C1"].model_axes()
    assert c1 == {
        "u_prior": "vamp",
        "init_prior_from_data": True,
        "freeze_prior_after_init": True,
        "hierarchy_mode": "legacy",
        "u_encoder_mode": "sample_conditioned",
        "scale_observations": False,
    }

    expected_changes = {
        "C2": {"hierarchy_mode": "centered_v2"},
        "C3": {
            "hierarchy_mode": "centered_v2",
            "scale_observations": True,
        },
        "C4": {
            "hierarchy_mode": "centered_v2",
            "u_encoder_mode": "sample_blind",
        },
    }
    for name, changes in expected_changes.items():
        observed = {
            key: value
            for key, value in candidates[name].model_axes().items()
            if c1[key] != value
        }
        assert observed == changes

    c0_changes = {
        key: value
        for key, value in candidates["C0"].model_axes().items()
        if c1[key] != value
    }
    assert c0_changes == {
        "u_prior": "mog",
        "init_prior_from_data": False,
        "freeze_prior_after_init": False,
    }
