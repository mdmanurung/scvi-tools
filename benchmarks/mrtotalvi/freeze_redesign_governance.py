"""Seal the explicit RDX-00 governance inventory."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .governance import (
    build_redesign_governance_manifest,
    write_redesign_governance_run,
)

BASE_COMMIT = "d8c8e997a67997a53f55923eb3ab14e6cf06f94c"
FROZEN_GOVERNANCE_INVENTORY = {
    "packet": [
        ".scratch/mrtotalvi-v2/PRD.md",
        ".scratch/mrtotalvi-v2/data-lineage-blocker.md",
        ".scratch/mrtotalvi-v2/historical-artifact-inventory.md",
        ".scratch/mrtotalvi-v2/improvement-roadmap-and-validation-protocol.md",
        ".scratch/mrtotalvi-v2/package-development-amendment.md",
        ".scratch/mrtotalvi-v2/package-verification.md",
        ".scratch/mrtotalvi-v2/scientific-validation-amendment-2026-07-26.md",
        ".scratch/mrtotalvi-v2/validation-report-2026-07-26.md",
        ".scratch/mrtotalvi-v2/issues/01-stage-0-foundation.md",
        ".scratch/mrtotalvi-v2/issues/02-stage-1-latent-feasibility.md",
        ".scratch/mrtotalvi-v2/issues/03-stage-2-counterfactual-api.md",
        ".scratch/mrtotalvi-v2/issues/03a-package-hardening.md",
        ".scratch/mrtotalvi-v2/issues/04-stage-3-benchmark-harness.md",
        ".scratch/mrtotalvi-v2/issues/05-stage-4-milo-bridge.md",
        ".scratch/mrtotalvi-v2/issues/06-stage-5-human-screen.md",
        ".scratch/mrtotalvi-v2/issues/07-stage-6-optional-de.md",
        ".scratch/mrtotalvi-v2/issues/08-stage-7-publication-validation.md",
        ".scratch/mrtotalvi-v2/issues/09-stage-8-release-hardening.md",
        ".scratch/mrtotalvi-v2-redesign/governance-amendment.md",
        "docs/adr/0007-mrtotalvi-v2-centered-counterfactuals.md",
        "docs/adr/0008-mrtotalvi-stable-latent-da-redesign.md",
        "docs/plans/2026-07-26-mrtotalvi-stable-latent-da-redesign.md",
        "docs/review-clear-execute/mrtotalvi-v2/handoff.md",
        "docs/review-clear-execute/mrtotalvi-v2/plan.md",
        "docs/review-clear-execute/mrtotalvi-v2/tasks.md",
        "docs/review-clear-execute/mrtotalvi-v2-redesign/handoff.md",
        "docs/review-clear-execute/mrtotalvi-v2-redesign/plan.md",
    ],
    "package_source": [
        "benchmarks/mrtotalvi/__init__.py",
        "benchmarks/mrtotalvi/aggregate.py",
        "benchmarks/mrtotalvi/config.py",
        "benchmarks/mrtotalvi/freeze_redesign_governance.py",
        "benchmarks/mrtotalvi/governance.py",
        "benchmarks/mrtotalvi/historical_comparator.py",
        "benchmarks/mrtotalvi/manifest.py",
        "benchmarks/mrtotalvi/metrics.py",
        "benchmarks/mrtotalvi/redesign_contract.py",
        "benchmarks/mrtotalvi/run_historical_comparator.py",
        "benchmarks/mrtotalvi/run_pilot.py",
        "benchmarks/mrtotalvi/runner.py",
        "benchmarks/mrtotalvi/simulation.py",
        "src/scvi/external/mrmultivi/_model.py",
        "src/scvi/external/mrmultivi/_module.py",
        "src/scvi/external/mrtotalvi/__init__.py",
        "src/scvi/external/mrtotalvi/_components.py",
        "src/scvi/external/mrtotalvi/_counterfactual.py",
        "src/scvi/external/mrtotalvi/_model.py",
        "src/scvi/external/mrtotalvi/_module.py",
        "src/scvi/external/mrtotalvi/_seed.py",
        "src/scvi/external/mrtotalvi/_stats.py",
    ],
    "regression_fixture": [
        "tests/benchmarks/mrtotalvi/test_aggregate.py",
        "tests/benchmarks/mrtotalvi/test_config.py",
        "tests/benchmarks/mrtotalvi/test_fixture.py",
        "tests/benchmarks/mrtotalvi/test_historical_comparator.py",
        "tests/benchmarks/mrtotalvi/test_manifest.py",
        "tests/benchmarks/mrtotalvi/test_metric_dictionary.py",
        "tests/benchmarks/mrtotalvi/test_metrics.py",
        "tests/benchmarks/mrtotalvi/test_redesign_contract.py",
        "tests/benchmarks/mrtotalvi/test_redesign_governance.py",
        "tests/benchmarks/mrtotalvi/test_simulation.py",
        "tests/external/mrmultivi/test_mrmultivi.py",
        "tests/external/mrtotalvi/legacy_oracle/d8c8e997/checksums.sha256",
        "tests/external/mrtotalvi/legacy_oracle/d8c8e997/environment_manifest.json",
        "tests/external/mrtotalvi/test_mrtotalvi.py",
        "tests/external/mrtotalvi/test_mrtotalvi_v2.py",
    ],
    "old_run": [
        ".scratch/mrtotalvi-v2/engineering-runs/"
        "20260725T223127Z-seed0-520ff544/checksums.sha256",
        ".scratch/mrtotalvi-v2/engineering-runs/"
        "20260725T233701Z-seed0-520ff544/checksums.sha256",
        ".scratch/mrtotalvi-v2/failed-oracle-attempt-20260725-2/"
        "checksums.sha256",
        ".scratch/mrtotalvi-v2/historical-comparator-runs/"
        "20260726T082206Z-7b6f4c6a-3f5ccbe4-a3839aa9/run-manifest.json",
        ".scratch/mrtotalvi-v2/historical-comparator-runs/"
        "20260726T083630Z-ebe540de-3f5ccbe4-a3839aa9/run-manifest.json",
        ".scratch/mrtotalvi-v2/historical-comparator-smokes/"
        "20260726T082104Z-7b6f4c6a-3bf2b746-a3839aa9/run-manifest.json",
        ".scratch/mrtotalvi-v2/validation-runs/"
        "20260726T075306Z-6fff0a0a-f7e179f9-b19fbf5d/run-manifest.json",
        ".scratch/mrtotalvi-v2/validation-runs/"
        "20260726T075604Z-2005fdab-faea410c-5f852a87/run-manifest.json",
        ".scratch/mrtotalvi-v2/validation-runs/"
        "20260726T075946Z-2005fdab-5a191c6e-e355e797/run-manifest.json",
    ],
    "baseline_evidence": [
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/baseline-report.md",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/baseline-results.json",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/checksums.sha256",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/environment.json",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/packet-hashes.sha256",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/prechange-status.txt",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/source-hashes.sha256",
        ".scratch/mrtotalvi-v2-redesign/executions/"
        "20260726T103620Z-baseline/zarr-probe.txt",
    ],
    "metric_dictionary": ["benchmarks/mrtotalvi/metric_dictionary.json"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            ".scratch/mrtotalvi-v2-redesign/governance-runs"
        ),
    )
    parser.add_argument(
        "--created-at",
        help="Optional timezone-aware ISO-8601 timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    """Build, verify, and atomically seal the frozen RDX-00 inventory."""
    args = _parse_args()
    created_at = (
        datetime.fromisoformat(args.created_at)
        if args.created_at
        else datetime.now(UTC)
    )
    manifest = build_redesign_governance_manifest(
        repository_root=args.repository_root,
        base_commit=BASE_COMMIT,
        created_at=created_at,
        inventory=FROZEN_GOVERNANCE_INVENTORY,
    )
    run_dir = write_redesign_governance_run(
        repository_root=args.repository_root,
        output_root=args.output_root,
        manifest=manifest,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
