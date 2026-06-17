"""Roll up multiseed benchmark JSON into PRD-style summaries."""

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


def summarize_multiseed(path: Path) -> dict[str, Any]:
    """Extract headline metrics from a ``run_multiseed`` JSON payload."""
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
    p2_conc = _get(summary, "b3.p2_concordance_vs_knn.agreement")
    if p1_f1:
        out["b3_p1_holdout_macro_f1"] = p1_f1
    if p2_conc:
        out["b3_p2_concordance"] = p2_conc

    return out


def merge_summaries(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"tasks": {}}
    for p in payloads:
        src = Path(p["source"]).name
        merged["tasks"][src] = {k: v for k, v in p.items() if k not in ("source",)}
    return merged


def main():
    ap = argparse.ArgumentParser(description="Summarize multiseed benchmark JSON files")
    ap.add_argument("inputs", nargs="+", type=Path, help="multiseed result JSON paths")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    summaries = [summarize_multiseed(p) for p in args.inputs]
    payload = merge_summaries(*summaries)
    payload["sources"] = [str(p) for p in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
