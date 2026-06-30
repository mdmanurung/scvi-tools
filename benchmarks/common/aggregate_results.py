"""Roll up benchmark JSON into PRD-style summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _get(d: dict, *keys: str, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarize_multiseed(path: Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract headline metrics from a ``run_multiseed`` JSON payload."""
    if data is None:
        data = json.loads(path.read_text())
    summary = data.get("summary", {})
    out: dict[str, Any] = {"source": str(path), "seeds": data.get("seeds")}

    for task, prefix in (("b1", "b1"), ("b2", "b2")):
        cyto_f1 = _get(summary, f"{prefix}.cytoanvi.macro_f1")
        knn_f1 = _get(summary, f"{prefix}.cytovi_knn.macro_f1")
        if cyto_f1:
            out[f"{task}_cytoanvi_macro_f1"] = cyto_f1
        if knn_f1:
            out[f"{task}_cytovi_knn_macro_f1"] = knn_f1
        if cyto_f1 and knn_f1:
            out[f"{task}_delta_macro_f1"] = {
                "mean": cyto_f1["mean"] - knn_f1["mean"],
                "std": (cyto_f1["std"] ** 2 + knn_f1["std"] ** 2) ** 0.5,
            }
        for model in ("cytoanvi", "cytovi"):
            bio = _get(summary, f"{prefix}.{model}.bio_conservation")
            batch = _get(summary, f"{prefix}.{model}.batch_correction")
            if bio:
                out[f"{task}_{model}_bio"] = bio
            if batch:
                out[f"{task}_{model}_batch"] = batch

    p1_f1 = _get(summary, "b3.p1_holdout.macro_f1")
    # New JSONs use p2_inter_method_agreement_vs_knn; old JSONs use p2_concordance_vs_knn.
    p2_conc = _get(summary, "b3.p2_inter_method_agreement_vs_knn.agreement") or _get(
        summary, "b3.p2_concordance_vs_knn.agreement"
    )
    if p1_f1:
        out["b3_p1_holdout_macro_f1"] = p1_f1
    if p2_conc:
        out["b3_p2_inter_method_agreement"] = p2_conc

    return out


def _summarize_single_task(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"task": payload.get("task", task)}
    if task == "b1":
        for model in ("cytoanvi", "cytovi_knn", "raw_marker_knn", "harmony_knn"):
            macro_f1 = _get(payload, model, "macro_f1")
            if macro_f1 is not None:
                out[f"{model}_macro_f1"] = macro_f1
    elif task == "b2":
        for model in ("cytoanvi", "cytovi"):
            for metric in ("bio_conservation", "batch_correction"):
                value = _get(payload, model, metric)
                if value is not None:
                    out[f"{model}_{metric}"] = value
    elif task == "b3":
        out["p1_holdout_macro_f1"] = _get(payload, "p1_holdout", "macro_f1")
        # New JSONs use p2_inter_method_agreement_vs_knn; old JSONs use p2_concordance_vs_knn.
        _new = _get(payload, "p2_inter_method_agreement_vs_knn", "agreement")
        _old = _get(payload, "p2_concordance_vs_knn", "agreement")
        out["p2_inter_method_agreement"] = _new if _new is not None else _old
    elif task == "b4":
        out["plain_replay_latent_drift"] = _get(
            payload, "plain_surgery", "replay_latent_drift"
        )
        out["continual_replay_latent_drift"] = _get(
            payload, "continual_update", "replay_latent_drift"
        )
        out["plain_query_macro_f1"] = _get(
            payload, "plain_surgery", "query_label_transfer", "macro_f1"
        )
        out["continual_query_macro_f1"] = _get(
            payload, "continual_update", "query_label_transfer", "macro_f1"
        )
    elif task == "b5":
        out["best_auroc"] = payload.get("best_auroc")
        out["mean_auroc"] = payload.get("mean_auroc")
        if "latent" in payload:
            out["latent_auroc"] = _get(payload, "latent", "auroc")
        if "logit" in payload:
            out["logit_auroc"] = _get(payload, "logit", "auroc")
    elif task == "b6":
        out["recommendation_status"] = payload.get("recommendation_status")
        if payload.get("recommended_lambda") is not None:
            out["recommended_lambda"] = payload.get("recommended_lambda")
            out["recommended_replay_latent_drift"] = payload.get(
                "recommended_replay_latent_drift"
            )
            out["recommended_query_macro_f1"] = payload.get("recommended_query_macro_f1")
    elif task == "b8":
        out["flat_macro_f1"] = _get(payload, "flat_ce", "macro_f1")
        out["hce_macro_f1"] = _get(payload, "hce_flat_predict", "macro_f1")
        out["hierarchical_macro_f1"] = _get(
            payload, "hce_hierarchical_predict", "macro_f1"
        )
        out["delta_hierarchical_vs_flat_macro_f1"] = payload.get(
            "delta_hierarchical_vs_flat_macro_f1",
            payload.get("delta_hce_vs_flat_macro_f1"),
        )
    elif task == "b9":
        out["status"] = payload.get("status")
        out["query_macro_f1"] = _get(payload, "query_label_transfer", "macro_f1")
        out["query_control_mapqc"] = payload.get("query_control_mapqc")
    return {k: v for k, v in out.items() if v is not None}


def summarize_result(path: Path) -> dict[str, Any]:
    """Extract headline metrics from one benchmark JSON payload."""
    data = json.loads(path.read_text())
    if "summary" in data:
        return summarize_multiseed(path, data)

    out: dict[str, Any] = {"source": str(path)}
    if "seed" in data:
        out["seed"] = data["seed"]
    if "dataset" in data:
        out["dataset"] = data["dataset"]

    for task in ("b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"):
        if isinstance(data.get(task), dict):
            out[task] = _summarize_single_task(task, data[task])
    if len(out) == 1 and isinstance(data.get("task"), str):
        task = data["task"].split("_", 1)[0]
        out[task] = _summarize_single_task(task, data)
    return out


def merge_summaries(*payloads: dict[str, Any]) -> dict[str, Any]:
    """Merge per-file summaries under filename keys."""
    merged: dict[str, Any] = {"tasks": {}}
    for p in payloads:
        src = Path(p["source"]).name
        merged["tasks"][src] = {k: v for k, v in p.items() if k not in ("source",)}
    return merged


def _inputs_from_dir(input_dir: Path, output: Path | None) -> list[Path]:
    paths = sorted(p for p in input_dir.rglob("*.json") if p.is_file())
    if output is not None:
        output = output.resolve()
        paths = [p for p in paths if p.resolve() != output]
    return paths


_MANIFEST_FIELDS = {
    "phase",
    "task",
    "path",
    "dataset",
    "seeds",
    "job_id",
    "status",
    "required",
}


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Publication manifest must contain a non-empty 'artifacts' list.")
    for idx, artifact in enumerate(artifacts):
        missing = sorted(_MANIFEST_FIELDS - set(artifact))
        if missing:
            raise ValueError(f"Manifest artifact {idx} missing required fields: {missing}")
        if not isinstance(artifact["seeds"], list):
            raise ValueError(f"Manifest artifact {idx} field 'seeds' must be a list.")
        artifact["required"] = bool(artifact["required"])
    return artifacts


def _validate_manifest_file(artifact: dict[str, Any], path: Path) -> None:
    payload = json.loads(path.read_text())
    if payload.get("dataset") != artifact["dataset"]:
        raise ValueError(
            f"{path} dataset={payload.get('dataset')!r} does not match manifest "
            f"dataset={artifact['dataset']!r}."
        )
    task = artifact["task"]
    if task not in payload and payload.get("task", "").split("_", 1)[0] != task:
        raise ValueError(f"{path} does not contain manifest task {task!r}.")
    expected_seeds = artifact["seeds"]
    if "seed" in payload and payload["seed"] not in expected_seeds:
        raise ValueError(
            f"{path} seed={payload['seed']!r} is not listed in manifest seeds={expected_seeds!r}."
        )
    if "seeds" in payload and sorted(payload["seeds"]) != sorted(expected_seeds):
        raise ValueError(
            f"{path} seeds={payload['seeds']!r} do not match manifest seeds={expected_seeds!r}."
        )


def _manifest_inputs(
    artifacts: list[dict[str, Any]],
    *,
    input_dir: Path | None,
    output: Path | None,
    parser: argparse.ArgumentParser,
) -> list[Path]:
    paths: list[Path] = []
    expected = {Path(a["path"]).resolve() for a in artifacts}
    if input_dir is not None:
        discovered = {p.resolve() for p in _inputs_from_dir(input_dir, output)}
        unknown = sorted(discovered - expected)
        if unknown:
            parser.error(
                "publication manifest mode rejects JSON files not listed in the manifest: "
                + ", ".join(str(p) for p in unknown)
            )
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.exists():
            if artifact["required"]:
                raise FileNotFoundError(f"required manifest artifact missing: {path}")
            continue
        if artifact["required"] and artifact["status"] != "complete":
            raise ValueError(
                f"required manifest artifact {path} has status={artifact['status']!r}; "
                "publication artifacts must be complete."
            )
        _validate_manifest_file(artifact, path)
        paths.append(path)
    return paths


def main():
    """Run the benchmark aggregation CLI."""
    ap = argparse.ArgumentParser(description="Summarize benchmark JSON files")
    ap.add_argument("inputs", nargs="*", type=Path, help="result JSON paths")
    ap.add_argument("--out", type=Path, default=None, help="output summary path")
    ap.add_argument("--input", type=Path, default=None, help="directory of result JSON files")
    ap.add_argument("--output", type=Path, default=None, help="output summary path")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="publication manifest listing exact expected benchmark result JSON files",
    )
    args = ap.parse_args()

    output = args.output or args.out
    if output is None:
        ap.error("one of --out or --output is required")

    manifest_artifacts = None
    if args.input is not None and not args.input.is_dir():
        ap.error(f"--input must be a directory: {args.input}")

    inputs = list(args.inputs)
    if args.manifest is not None:
        manifest_artifacts = _load_manifest(args.manifest)
        inputs.extend(
            _manifest_inputs(
                manifest_artifacts,
                input_dir=args.input,
                output=output,
                parser=ap,
            )
        )
    elif args.input is not None:
        inputs.extend(_inputs_from_dir(args.input, output))
    if not inputs:
        ap.error("at least one input JSON file or --input directory is required")

    summaries = [summarize_result(p) for p in inputs]
    payload = merge_summaries(*summaries)
    payload["aggregation_mode"] = (
        "publication_manifest" if args.manifest is not None else "exploratory_recursive"
    )
    payload["sources"] = [str(p) for p in inputs]
    if manifest_artifacts is not None:
        payload["manifest_artifacts"] = manifest_artifacts
    payload["missing_optional"] = [
        task
        for task in ("b4", "b6", "b8", "b9")
        if not any(task in summary for summary in summaries)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
