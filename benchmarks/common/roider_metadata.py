"""Patient-level metadata for the full Roider BNHL cohort."""

from __future__ import annotations

import json
from pathlib import Path

_REPO_DATA = Path(__file__).resolve().parents[2] / "data"
ENTITY_JSON = _REPO_DATA / "roider_full" / "patient_entity.json"
LEIDEN_CACHE_DIR = _REPO_DATA / "roider_full"
VIGNETTE_PANEL1 = _REPO_DATA / "Roider_et_al_BNHL_panel1.h5ad"
SUPPLEMENTARY_XLSX = _REPO_DATA / "41556_2024_1358_MOESM3_ESM.xlsx"

# Paper uses DLBCL GCB / non-GCB; vignette ``Entity`` column collapses to ``DLBCL``.
_ENTITY_ALIASES = {
    "DLBCL, GCB": "DLBCL",
    "DLBCL, non-GCB": "DLBCL",
}


def normalize_entity(raw: str) -> str:
    """Map supplementary-table entity strings to vignette-compatible labels."""
    text = str(raw).strip()
    return _ENTITY_ALIASES.get(text, text)


def entity_map_from_supplementary(xlsx: Path | None = None) -> dict[str, str]:
    """PatientID → disease entity from Nature Supplementary Table 1 (MOESM3 workbook)."""
    import pandas as pd

    path = xlsx or SUPPLEMENTARY_XLSX
    if not path.exists():
        raise FileNotFoundError(f"supplementary workbook missing at {path}")
    df = pd.read_excel(path, sheet_name="Supplementary Table 1")
    if "PatientID" not in df.columns or "Entity" not in df.columns:
        raise KeyError("Supplementary Table 1 must contain PatientID and Entity columns")
    out: dict[str, str] = {}
    for pid, entity in zip(df["PatientID"], df["Entity"], strict=False):
        pid = str(pid).strip()
        if not pid or pid.lower() == "nan":
            continue
        out[pid] = normalize_entity(entity)
    return out


def entity_map_from_vignette(vignette_h5ad: Path | None = None) -> dict[str, str]:
    """PatientID → disease entity from the CytoVI tutorial ``.h5ad`` (33 of 63 patients)."""
    import scanpy as sc

    path = vignette_h5ad or VIGNETTE_PANEL1
    if not path.exists():
        raise FileNotFoundError(f"vignette panel1 missing at {path}")
    adata = sc.read(path)
    if "Entity" not in adata.obs or "PatientID" not in adata.obs:
        raise KeyError("vignette .h5ad must contain PatientID and Entity columns")
    return (
        adata.obs.groupby("PatientID", observed=True)["Entity"]
        .first()
        .astype(str)
        .to_dict()
    )


def load_patient_entity_map(
    vignette_h5ad: Path | None = None,
    supplementary_xlsx: Path | None = None,
    *,
    refresh: bool = False,
) -> dict[str, str]:
    """Load cached JSON or rebuild from supplementary table (preferred) + vignette fallback."""
    if ENTITY_JSON.exists() and not refresh:
        return json.loads(ENTITY_JSON.read_text())

    mapping: dict[str, str] = {}
    supp = supplementary_xlsx or SUPPLEMENTARY_XLSX
    if supp.exists():
        mapping.update(entity_map_from_supplementary(supp))
    if VIGNETTE_PANEL1.exists() or vignette_h5ad:
        for pid, entity in entity_map_from_vignette(vignette_h5ad).items():
            mapping.setdefault(pid, entity)

    if not mapping:
        raise FileNotFoundError(
            f"No entity metadata found; place {SUPPLEMENTARY_XLSX.name} under data/ "
            "or provide vignette panel1 .h5ad"
        )

    ENTITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENTITY_JSON.write_text(json.dumps(dict(sorted(mapping.items())), indent=2))
    return mapping


def _leiden_cache_path(labels_key: str, resolution: float) -> Path:
    slug = labels_key.replace(" ", "_")
    return LEIDEN_CACHE_DIR / f"panel1_{slug}_leiden_r{resolution}.parquet"


def apply_leiden_cell_types(
    p1,
    *,
    labels_key: str = "cell_type",
    resolution: float = 1.0,
    refresh: bool = False,
    write_cache: bool = True,
    seed: int = 0,
) -> int:
    """Leiden clusters on panel 1 as proxy ``cell_type`` labels; cached by resolution."""
    from benchmarks.cytoanvi.data import _leiden_labels

    cache = _leiden_cache_path(labels_key, resolution)
    if cache.exists() and not refresh:
        import pandas as pd

        table = pd.read_parquet(cache)
        if labels_key not in table.columns:
            raise KeyError(f"{cache} missing column {labels_key!r}")
        labels = table[labels_key].reindex(p1.obs_names)
        if labels.isna().any():
            _leiden_labels(p1, labels_key=labels_key, resolution=resolution, seed=seed)
        else:
            p1.obs[labels_key] = labels.astype(str).to_numpy()
            return int(p1.obs[labels_key].nunique())

    _leiden_labels(p1, labels_key=labels_key, resolution=resolution, seed=seed)
    if write_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        p1.obs[[labels_key]].to_parquet(cache)
    return int(p1.obs[labels_key].nunique())


def annotate_roider_obs(
    merged,
    p1,
    p2,
    entity_map: dict[str, str],
    *,
    labels_key: str = "cell_type",
    leiden: bool = True,
    leiden_resolution: float = 1.0,
    leiden_refresh: bool = False,
    leiden_write_cache: bool = True,
    seed: int = 0,
):
    """Add ``Entity``, ``batch``, and Leiden proxy labels on panel 1."""
    for adata in (merged, p1, p2):
        pid = adata.obs["PatientID"].astype(str)
        adata.obs["Entity"] = pid.map(entity_map).fillna("Unknown")
        adata.obs["batch"] = adata.obs["patient_batch"].astype(str)
    if leiden:
        apply_leiden_cell_types(
            p1,
            labels_key=labels_key,
            resolution=leiden_resolution,
            refresh=leiden_refresh,
            write_cache=leiden_write_cache,
            seed=seed,
        )
        # Panel 1 only (matches vignette); panel-2 cells stay unlabeled for B3 query side.
        merged.obs[labels_key] = "Unknown"
        merged.obs.loc[p1.obs_names, labels_key] = p1.obs[labels_key].astype(str).values
        p2.obs[labels_key] = "Unknown"


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Roider full-cohort metadata utilities")
    ap.add_argument("--refresh-entity", action="store_true", help="Rebuild patient_entity.json")
    ap.add_argument("--refresh-leiden", action="store_true", help="Recompute panel1 Leiden labels")
    ap.add_argument("--leiden-resolution", type=float, default=1.0)
    ap.add_argument("--labels-key", default="cell_type")
    ap.add_argument("--supplementary", type=Path, default=SUPPLEMENTARY_XLSX)
    args = ap.parse_args()

    if args.refresh_entity:
        mapping = load_patient_entity_map(supplementary_xlsx=args.supplementary, refresh=True)
        from collections import Counter

        print(f"wrote {ENTITY_JSON} ({len(mapping)} patients)")
        print(Counter(mapping.values()))

    if args.refresh_leiden:
        from benchmarks.cytoanvi.data import load_roider_full

        _, p1, _ = load_roider_full(leiden_labels=False)
        n = apply_leiden_cell_types(
            p1,
            labels_key=args.labels_key,
            resolution=args.leiden_resolution,
            refresh=True,
        )
        print(f"wrote {_leiden_cache_path(args.labels_key, args.leiden_resolution)} ({n} clusters)")
    elif not (args.refresh_entity or args.refresh_leiden):
        ap.print_help()


if __name__ == "__main__":
    main()
