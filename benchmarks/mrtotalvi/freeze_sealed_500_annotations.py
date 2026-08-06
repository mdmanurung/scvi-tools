"""Freeze the exact ordered sealed-500 evaluation labels without replacement."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import anndata as ad

from .convergence_runner import (
    SEALED_500_ANNOTATION_SOURCE,
    SEALED_500_ANNOTATION_SOURCE_SHA256,
    SEALED_500_H5AD,
    SEALED_500_SHA256,
    SEALED_500_STATE_ANNOTATION_SHA256,
    SEALED_500_STATE_ANNOTATIONS,
    _sha256_file,
    _state_annotation_digest,
)
from .historical_comparator import read_selected_categorical
from .human_lineage import sha256_lines

SCHEMA_VERSION = "mrtotalvi-sealed-500-state-annotations-v1"


def _regular_exact(path: Path, *, name: str, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file.")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 drifted.")


def build_payload(*, repository_root: Path) -> dict[str, object]:
    """Read the exact authority once and return its compact ordered closure."""
    root = repository_root.resolve()
    fixture_path = root / SEALED_500_H5AD
    source_path = SEALED_500_ANNOTATION_SOURCE
    _regular_exact(
        fixture_path,
        name="Sealed 500-cell fixture",
        expected_sha256=SEALED_500_SHA256,
    )
    _regular_exact(
        source_path,
        name="Sealed 500-cell annotation source",
        expected_sha256=SEALED_500_ANNOTATION_SOURCE_SHA256,
    )

    fixture = ad.read_h5ad(fixture_path, backed="r")
    try:
        if fixture.shape != (500, 1000):
            raise ValueError("Sealed 500-cell fixture dimensions drifted.")
        cell_ids = tuple(str(value) for value in fixture.obs_names)
    finally:
        fixture.file.close()
    if len(cell_ids) != 500 or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Sealed 500-cell fixture IDs drifted.")

    labels_array = read_selected_categorical(
        source_path,
        selected_cell_ids=cell_ids,
        column="cell_label_l1p5",
    )
    labels = tuple(str(value) for value in labels_array)
    if _state_annotation_digest(labels) != SEALED_500_STATE_ANNOTATION_SHA256:
        raise ValueError("Sealed 500-cell state annotations drifted.")
    records = [
        {"cell_id": cell_id, "label": label}
        for cell_id, label in zip(cell_ids, labels, strict=True)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture": {
            "fixture_id": "sealed_500",
            "h5ad_path": SEALED_500_H5AD.as_posix(),
            "h5ad_sha256": SEALED_500_SHA256,
            "ordered_cell_ids_sha256": sha256_lines(cell_ids),
        },
        "annotation": {
            "column": "cell_label_l1p5",
            "source_h5ad_path": str(source_path),
            "source_h5ad_sha256": SEALED_500_ANNOTATION_SOURCE_SHA256,
            "ordered_labels_sha256": sha256_lines(labels),
            "ordered_selection_sha256": sha256_lines(
                tuple(
                    f"{cell_id}\t{label}"
                    for cell_id, label in zip(cell_ids, labels, strict=True)
                )
            ),
        },
        "records": records,
    }


def write_payload_no_replace(path: Path, payload: object) -> None:
    """Publish deterministic JSON with an atomic hard-link no-replace gate."""
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
        # The destination was created by ``os.link`` from our regular temp file,
        # so a following chmod cannot traverse an attacker-controlled symlink.
        os.chmod(output, 0o444)
        directory_fd = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    """Freeze the default payload or one explicitly requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SEALED_500_STATE_ANNOTATIONS,
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    payload = build_payload(repository_root=repository_root)
    write_payload_no_replace(repository_root / args.output, payload)
    output = (repository_root / args.output).resolve()
    print(json.dumps({"path": str(output), "sha256": _sha256_file(output)}))


if __name__ == "__main__":
    main()
