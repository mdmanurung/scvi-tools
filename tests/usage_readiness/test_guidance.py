from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

PRIMARY_GUIDANCE = (
    "README.md",
    "benchmarks/cytoanvi/README.md",
    "docs/installation.md",
    "docs/api/user.md",
    "docs/user_guide/models/cytoanvi.md",
    "docs/user_guide/models/mr_multimodal.md",
    "docs/tutorials/parameter_selection/CytoANVI_parameter_selection.md",
    "docs/tutorials/parameter_selection/MrTotalVI_parameter_selection.md",
    "vignettes/cytoanvi_example_reference_query.py",
    "vignettes/cytoanvi_showcase.py",
    "vignettes/cytoanvi_treearches_synthetic.py",
)


@pytest.mark.parametrize("relative", PRIMARY_GUIDANCE)
def test_primary_guidance_links_authoritative_usage_readiness(relative: str) -> None:
    text = (ROOT / relative).read_text()
    assert "usage_readiness" in text, f"{relative} does not link the authoritative status surface"


def test_root_owned_tutorial_paths_exist() -> None:
    for relative in (
        "vignettes/cytoanvi_example_reference_query.py",
        "vignettes/cytoanvi_treearches_synthetic.py",
        "docs/review-clear-execute/cytoanvi-mrtotalvi-usage-readiness/"
        "external-submodule-tutorial-handoff.md",
    ):
        assert (ROOT / relative).is_file()


def test_usage_readiness_ci_validates_without_building_candidate() -> None:
    workflow = (ROOT / ".github/workflows/usage_readiness_contracts.yml").read_text()
    assert "python scripts/validate_usage_readiness.py" in workflow
    assert "--confcutdir=tests/usage_readiness tests/usage_readiness" in workflow
    assert "python -m build" not in workflow
    assert "scripts/accept_usage_readiness_wheel --wheel" not in workflow


def test_guidance_does_not_advertise_quarantined_or_external_paths() -> None:
    text = "\n".join((ROOT / relative).read_text() for relative in PRIMARY_GUIDANCE)
    forbidden = (
        "model.get_uncertainty(",
        "select_replay_by_uncertainty(model",
        "Registering labels_key silently",
        "sample-blind base `u`",
        "route differential abundance through MrMultiVI",
        "Cross-validate eps-space LFC",
        "docs/tutorials/notebooks/cytometry/cytoanvi_treearches_synthetic.py",
        "python docs/tutorials/notebooks/cytometry/cytoanvi_example_reference_query.py",
        "task_mod.task_b5_novelty(",
        "task_mod.task_b4_continual(",
        "get_uncertainty, Bregman info",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_mrtotalvi_guidance_states_fail_closed_boundaries() -> None:
    guide = (ROOT / "docs/user_guide/models/mr_multimodal.md").read_text()
    parameter_guide = (
        ROOT / "docs/tutorials/parameter_selection/MrTotalVI_parameter_selection.md"
    ).read_text()
    combined = guide + parameter_guide
    for phrase in (
        "raw counts",
        'u_prior_supervision="none"',
        "donor-pseudobulk",
        "use_vmap=True",
        "descriptive",
        "streaming",
        "new-sample inference",
    ):
        assert phrase in combined


def test_cytoanvi_guidance_states_training_boundary_and_no_go() -> None:
    parameter_guide = (
        ROOT / "docs/tutorials/parameter_selection/CytoANVI_parameter_selection.md"
    ).read_text()
    assert "actual training split" in parameter_guide
    assert "Stable `get_uncertainty()`" in parameter_guide
    assert "reference replay set" in parameter_guide
    assert "external matched control" in parameter_guide


def test_cytoanvi_benchmark_dependency_names_candidate_distribution() -> None:
    tasks = (ROOT / "benchmarks/cytoanvi/tasks.py").read_text()
    assert "cytoanvi[cytoanvi-mapping-qc]" in tasks
    assert "scvi-tools[cytoanvi-mapping-qc]" not in tasks
