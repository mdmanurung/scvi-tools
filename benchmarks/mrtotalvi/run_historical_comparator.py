"""Run a bounded C0-C4 sensitivity on the sealed 500-cell human fixture.

This is engineering evidence on a historical comparator. It is not canonical,
not QC-pass, not biological validation, and not model-promotion evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import candidate_configs
from .historical_comparator import (
    ALLOWED_ENGINEERING_ANNOTATIONS,
    HistoricalRunConfig,
    aggregate_historical_results,
    read_selected_categorical,
    run_historical_candidate,
)
from .manifest import (
    ArtifactRecord,
    RunManifest,
    make_run_id,
    sha256_file,
    verify_run_manifest,
)
from .run_pilot import (
    _code_manifest,
    _environment_manifest,
    _parse_csv,
    _parse_seeds,
    _payload_digest,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / ".scratch"
    / "mrtotalvi-v2"
    / "engineering-runs"
    / "20260725T233701Z-seed0-520ff544"
    / "checkpoint"
    / "adata.h5ad"
)
DEFAULT_SOURCE = Path(
    "/exports/para-lipg-hpc/mdmanurung/schisto_citeseq/"
    "analysis/harmonized_integration/outputs/human_immune_joint.h5ad"
)
EXPECTED_FIXTURE_SHA256 = (
    "b63a7df6b57d4db5bf0ce9e091ca36db9d19ad7c6ea798c7224a52f3a7d51dff"
)
RECORDED_SOURCE_SHA256 = (
    "520ff544daae6192efd7f3501669e05b0122e6fbaf8e9de0246122cecd1de2da"
)
SCIENTIFIC_SCOPE = (
    "historical comparator; not canonical; not QC-pass; not biological "
    "validation; not promotion evidence"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-h5ad", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--annotation-source-h5ad", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--state-annotation",
        choices=ALLOWED_ENGINEERING_ANNOTATIONS,
        default="cell_label_l2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".scratch/mrtotalvi-v2/historical-comparator-runs"),
    )
    parser.add_argument(
        "--candidates",
        type=_parse_csv,
        default=tuple(candidate_configs()),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-latent", type=int, default=20)
    parser.add_argument("--n-prior-components", type=int, default=20)
    return parser.parse_args()


def _sha256_lines(values: list[str] | tuple[str, ...] | np.ndarray) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _artifact_records(run_dir: Path) -> tuple[ArtifactRecord, ...]:
    return tuple(
        ArtifactRecord(
            path=path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "run-manifest.json"
    )


def _seal_manifest(
    run_dir: Path,
    *,
    run_id: str,
    created_at: datetime,
    code_digest: str,
    config_digest: str,
    data_digest: str,
    status: str,
) -> None:
    manifest = RunManifest(
        schema_version="mrtotalvi-benchmark-run-v1",
        run_id=run_id,
        created_at=created_at.isoformat(),
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
        evidence_tier="pilot_cache",
        scientific_scope=SCIENTIFIC_SCOPE,
        status=status,
        artifacts=_artifact_records(run_dir),
    )
    _write_json(run_dir / "run-manifest.json", manifest.to_dict())
    verify_run_manifest(run_dir / "run-manifest.json")


def _assessment(
    aggregate: dict,
    *,
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict:
    per_candidate = aggregate["candidates"]
    hypotheses = {}
    centered = [
        per_candidate[candidate]["per_seed"][str(seed)]["metrics"][
            "centering_max_abs"
        ]
        for candidate in candidates
        if candidate in {"C2", "C3", "C4"}
        for seed in seeds
    ]
    if centered:
        hypotheses["H1_centering"] = {
            "verdict": (
                "engineering-feasible"
                if all(value <= 1e-6 for value in centered)
                else "mechanism-not-supported"
            ),
            "maximum_absolute_error": max(centered),
        }

    if {"C2", "C4"}.issubset(candidates):
        leakage_deltas = [
            per_candidate["C4"]["per_seed"][str(seed)]["metrics"][
                "sample_balanced_accuracy_within_state"
            ]
            - per_candidate["C2"]["per_seed"][str(seed)]["metrics"][
                "sample_balanced_accuracy_within_state"
            ]
            for seed in seeds
        ]
        hypotheses["H3_sample_blind_leakage_sensitivity"] = {
            "verdict": "inconclusive",
            "direction_consistent": all(delta <= 0.0 for delta in leakage_deltas),
            "per_seed_C4_minus_C2": leakage_deltas,
            "lower_is_better": True,
            "limitation": (
                "historical noncanonical annotation sensitivity; not proof of "
                "biological sample neutrality"
            ),
        }

    if {"C1", "C4"}.issubset(candidates):
        predictive_degradation = []
        state_deltas = []
        for seed in seeds:
            c1_metrics = per_candidate["C1"]["per_seed"][str(seed)]["metrics"]
            c4_metrics = per_candidate["C4"]["per_seed"][str(seed)]["metrics"]
            predictive_degradation.append(
                max(
                    0.0,
                    (c1_metrics["heldout_elbo"] - c4_metrics["heldout_elbo"])
                    / abs(c1_metrics["heldout_elbo"]),
                )
            )
            state_deltas.append(
                c4_metrics["state_balanced_accuracy"]
                - c1_metrics["state_balanced_accuracy"]
            )
        hypotheses["H4_preservation_sensitivity"] = {
            "verdict": "inconclusive",
            "numeric_reference_compatible": (
                all(value <= 0.02 for value in predictive_degradation)
                and all(value >= -0.02 for value in state_deltas)
            ),
            "per_seed_relative_ELBO_degradation_C4_vs_C1": predictive_degradation,
            "per_seed_state_balanced_accuracy_C4_minus_C1": state_deltas,
            "limitation": (
                "numeric references are from the frozen fast-screen gates, "
                "which cannot pass on this noncanonical 500-cell fixture"
            ),
        }

    if {"C2", "C3"}.issubset(candidates):
        deltas = {}
        metric_names = (
            "heldout_elbo",
            "heldout_sample_elbo_sd",
            "state_balanced_accuracy",
            "sample_balanced_accuracy_within_state",
            "knn_state_accuracy",
        )
        for metric in metric_names:
            deltas[metric] = [
                per_candidate["C3"]["per_seed"][str(seed)]["metrics"][metric]
                - per_candidate["C2"]["per_seed"][str(seed)]["metrics"][metric]
                for seed in seeds
            ]
        hypotheses["H5_balanced_weighting_negative_control"] = {
            "verdict": (
                "engineering-feasible"
                if all(
                    abs(value) <= 1e-8
                    for metric_deltas in deltas.values()
                    for value in metric_deltas
                )
                else "inconclusive"
            ),
            "per_seed_C3_minus_C2": deltas,
            "expected": (
                "C2 and C3 are numerically identical because the fixture has "
                "exactly 25 cells in every registered sample"
            ),
        }

    return {
        "schema_version": "mrtotalvi-historical-comparator-assessment-v1",
        "hypotheses": hypotheses,
        "overall_verdict": "inconclusive",
        "selection": "none",
        "promotion": "prohibited",
        "limitations": [
            "historical comparator, not a canonical or QC-pass cohort",
            "500 hash-selected cells and a bounded three-epoch CPU fit",
            "cell-state annotations were read only for descriptive evaluation",
            "no Milo DA, DE, causal, or external-validation endpoint",
            "no candidate or default selection",
        ],
    }


def run_historical_comparator(args: argparse.Namespace) -> Path:
    """Execute and atomically seal one bounded historical comparison."""
    candidates = tuple(args.candidates)
    unknown = sorted(set(candidates) - set(candidate_configs()))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}.")
    seeds = tuple(args.seeds)
    fixture_path = args.fixture_h5ad.resolve()
    fixture_sha256 = sha256_file(fixture_path)
    if fixture_sha256 != EXPECTED_FIXTURE_SHA256:
        raise ValueError(
            f"Fixture SHA-256 is {fixture_sha256}, expected "
            f"{EXPECTED_FIXTURE_SHA256}."
        )

    import anndata as ad

    fixture_header = ad.read_h5ad(fixture_path, backed="r")
    selected_cell_ids = tuple(fixture_header.obs_names.astype(str))
    fixture_header.file.close()
    state_labels = read_selected_categorical(
        args.annotation_source_h5ad,
        selected_cell_ids=selected_cell_ids,
        column=args.state_annotation,
    )
    annotation_digest = _sha256_lines(state_labels)
    ordered_cell_digest = _sha256_lines(selected_cell_ids)
    execution_config = {
        "fixture_h5ad": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "annotation_source_h5ad": str(args.annotation_source_h5ad.resolve()),
        "annotation_source_recorded_sha256": RECORDED_SOURCE_SHA256,
        "annotation_source_hash_recomputed": False,
        "state_annotation": args.state_annotation,
        "state_annotation_sha256": annotation_digest,
        "ordered_cell_sha256": ordered_cell_digest,
        "qc_field_accessed": False,
        "candidates": list(candidates),
        "seeds": list(seeds),
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "n_latent": args.n_latent,
        "n_prior_components": args.n_prior_components,
        "candidate_axes": {
            name: candidate_configs()[name].model_axes()
            for name in candidates
        },
        "scientific_scope": SCIENTIFIC_SCOPE,
    }
    code_manifest = _code_manifest()
    code_digest = code_manifest["code_digest"]
    config_digest = _payload_digest(execution_config)
    data_digest = _payload_digest(
        {
            "fixture_sha256": fixture_sha256,
            "ordered_cell_sha256": ordered_cell_digest,
            "state_annotation": args.state_annotation,
            "state_annotation_sha256": annotation_digest,
        }
    )
    created_at = datetime.now(UTC).replace(microsecond=0)
    run_id = make_run_id(
        timestamp=created_at,
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    failed_destination = output_root / f"{run_id}-failed"
    temporary = output_root / f".{run_id}.tmp-{os.getpid()}"
    for path in (destination, failed_destination, temporary):
        if path.exists():
            raise FileExistsError(f"Refusing existing benchmark destination {path}.")
    temporary.mkdir()

    try:
        _write_json(temporary / "configuration.json", execution_config)
        _write_json(temporary / "code-manifest.json", code_manifest)
        _write_json(temporary / "environment.json", _environment_manifest())
        unique_states, state_counts = np.unique(state_labels, return_counts=True)
        _write_json(
            temporary / "annotation-manifest.json",
            {
                "state_annotation": args.state_annotation,
                "state_annotation_sha256": annotation_digest,
                "ordered_cell_sha256": ordered_cell_digest,
                "n_cells": len(selected_cell_ids),
                "n_states": len(unique_states),
                "state_counts": {
                    str(state): int(count)
                    for state, count in zip(
                        unique_states,
                        state_counts,
                        strict=True,
                    )
                },
                "qc_field_accessed": False,
                "scientific_scope": SCIENTIFIC_SCOPE,
            },
        )

        results = []
        representations = {}
        for candidate in candidates:
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "event": "run_start",
                            "candidate": candidate,
                            "seed": seed,
                            "scope": SCIENTIFIC_SCOPE,
                        }
                    ),
                    flush=True,
                )
                result, representation = run_historical_candidate(
                    fixture_path,
                    state_labels=state_labels,
                    fixture_sha256=fixture_sha256,
                    config=HistoricalRunConfig(
                        candidate=candidate,
                        seed=seed,
                        max_epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        n_latent=args.n_latent,
                        n_prior_components=args.n_prior_components,
                    ),
                )
                results.append(result)
                representations[(candidate, seed)] = representation
                _write_json(
                    temporary / "results" / f"{candidate}-seed{seed}.json",
                    result,
                )
                representation_path = (
                    temporary
                    / "representations"
                    / f"{candidate}-seed{seed}.npz"
                )
                representation_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    representation_path,
                    u=representation,
                    cell_name=np.asarray(selected_cell_ids, dtype=str),
                )
                print(
                    json.dumps(
                        {
                            "event": "run_complete",
                            "candidate": candidate,
                            "seed": seed,
                            "wall_seconds": result["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
                gc.collect()

        aggregate = aggregate_historical_results(
            results,
            representations=representations,
            expected_candidates=candidates,
            expected_seeds=seeds,
            fixture_sha256=fixture_sha256,
            k=15,
        )
        _write_json(temporary / "aggregate.json", aggregate)
        _write_json(
            temporary / "assessment.json",
            _assessment(
                aggregate,
                candidates=candidates,
                seeds=seeds,
            ),
        )
        _seal_manifest(
            temporary,
            run_id=run_id,
            created_at=created_at,
            code_digest=code_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            status="complete",
        )
        temporary.rename(destination)
        verify_run_manifest(destination / "run-manifest.json")
        print(json.dumps({"event": "sealed", "path": str(destination)}), flush=True)
        return destination
    except BaseException as error:
        _write_json(
            temporary / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            _seal_manifest(
                temporary,
                run_id=run_id,
                created_at=created_at,
                code_digest=code_digest,
                config_digest=config_digest,
                data_digest=data_digest,
                status="failed",
            )
        except (OSError, TypeError, ValueError) as seal_error:
            _write_json(
                temporary / "manifest-seal-failure.json",
                {
                    "error_type": type(seal_error).__name__,
                    "error": str(seal_error),
                    "traceback": traceback.format_exc(),
                },
            )
        temporary.rename(failed_destination)
        raise


def main() -> None:
    """CLI entrypoint."""
    run_historical_comparator(_parse_args())


if __name__ == "__main__":
    main()
