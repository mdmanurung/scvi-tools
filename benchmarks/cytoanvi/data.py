"""Dataset loaders for the CytoANVI benchmark (the two CytoVI vignette datasets).

D1 — Roider BNHL (CytoVI advanced tutorial): two antibody panels, shared backbone, labels in
     panel 1 only. Preprocessed .h5ad (arcsinh + scaled + subsampled). No FCS reader needed.
D2 — Nuñez PBMC (CytoVI batch tutorial): single panel, fully labelled, two batches. Raw .fcs —
     needs an FCS reader (readfcs/flowio) + cytovi preprocessing.

Figshare returns HTTP 202 ("file generating, retry") for these ids; download() retries. If the
sandbox blocks egress, fetch the files yourself (see README) and point --data-dir at them.

A synthetic loader (make_synthetic_panels) backs the --smoke mode so the harness is runnable
without any download.
"""

from __future__ import annotations

import os
import time
import urllib.request

import numpy as np

# Single source for Figshare IDs and layer constants — imported from the common modules.
# data.py re-exports them so callers that historically imported from here still work.
from benchmarks.common.fetch_data import VIGNETTE as _VIGNETTE
from benchmarks.common.training import NAN_LAYER, SCALED_LAYER  # noqa: F401 — re-export

FIGSHARE = {name: fig_id for name, (fig_id, _) in _VIGNETTE.items()}
FIGSHARE.update(
    {
        "roider_p1.h5ad": FIGSHARE["Roider_et_al_BNHL_panel1.h5ad"],
        "roider_p2.h5ad": FIGSHARE["Roider_et_al_BNHL_panel2.h5ad"],
    }
)

ROIDER_PANEL_ALIASES = {
    "panel1": ("Roider_et_al_BNHL_panel1.h5ad", "roider_p1.h5ad"),
    "panel2": ("Roider_et_al_BNHL_panel2.h5ad", "roider_p2.h5ad"),
}

# Entity split for B4/B6 continual-update tasks using the Roider BNHL cohort.
# FL+DLBCL are used as the reference atlas; MCL+rLN are the novel query entities added in Phase 2.
# Requires ``adata.obs["Entity"]`` populated by
# ``benchmarks.common.roider_metadata.annotate_roider_obs``.
BNHL_CONTINUAL_SPLIT: dict[str, list[str]] = {
    "ref": ["FL", "DLBCL"],
    "query": ["MCL", "rLN"],
}

# Repo-root ``data/`` (sibling of ``benchmarks/``) — common drop location for downloaded files.
_REPO_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
)


def _resolve_file(data_dir: str, name: str) -> str:
    """Find a benchmark data file in ``data_dir`` or repo ``data/``."""
    for base in (data_dir, _REPO_DATA):
        p = os.path.join(base, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return os.path.join(data_dir, name)


def _resolve_any_file(data_dir: str, names: tuple[str, ...]) -> str | None:
    """Find the first existing file among a set of canonical/legacy names."""
    for name in names:
        p = _resolve_file(data_dir, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def holdout_safe_name(holdout_type: str) -> str:
    """Filesystem-safe slug for B5 sweep JSON filenames."""
    return (
        holdout_type.replace("+", "_plus")
        .replace("-", "_minus")
        .replace(" ", "_")
    )


def download(name: str, data_dir: str, retries: int = 8, wait: float = 15.0) -> str:
    """Download a known Figshare file into ``data_dir`` (retries the 202 'generating' state)."""
    if name not in FIGSHARE:
        raise KeyError(f"unknown file {name!r}; known: {list(FIGSHARE)}")
    dest = os.path.join(data_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(data_dir, exist_ok=True)
    url = f"https://figshare.com/ndownloader/files/{FIGSHARE[name]}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                if r.status == 200:
                    data = r.read()
                    if data:
                        with open(dest, "wb") as fh:
                            fh.write(data)
                        return dest
        except Exception as e:  # noqa: BLE001 — surface and retry transient network errors
            print(f"  [download] {name} attempt {attempt}: {type(e).__name__}: {e}")
        else:
            print(f"  [download] {name} attempt {attempt}: HTTP {r.status} (generating), retrying")
        time.sleep(wait)
    raise RuntimeError(
        f"Could not download {name} from Figshare after {retries} attempts (202 'generating' or "
        f"blocked egress). Fetch it manually: curl -L -o {dest} {url}"
    )


def _ensure_scaled(adata):
    """Guarantee a 'scaled' layer (CytoVI reads it); fall back to X if the .h5ad only has X."""
    if SCALED_LAYER not in adata.layers:
        adata.layers[SCALED_LAYER] = adata.X.copy()
    return adata


def load_roider(data_dir: str, auto_download: bool = True):
    """Load D1: merge the two Roider panels into one backbone/panel-specific AnnData.

    Returns ``(merged, p1, p2)``. ``merged`` has the ``_nan_mask`` from
    :func:`~scvi.external.cytovi.merge_batches`; panel 1 carries the cell-type labels.
    """
    import scanpy as sc

    from scvi.external import cytovi

    paths = {}
    for panel_key, names in ROIDER_PANEL_ALIASES.items():
        p = _resolve_any_file(data_dir, names)
        if p is None:
            if not auto_download:
                raise FileNotFoundError(
                    f"{names[0]} missing; run with auto-download or fetch it (README)."
                )
            p = download(names[0], data_dir)
        paths[panel_key] = p

    p1 = _ensure_scaled(sc.read(paths["panel1"]))
    p2 = _ensure_scaled(sc.read(paths["panel2"]))
    merged = cytovi.merge_batches([p1, p2], batch_key="panel_batch")
    return merged, p1, p2


def _leiden_labels(
    adata,
    labels_key: str = "cell_type",
    resolution: float = 1.0,
    layer: str = SCALED_LAYER,
    seed: int = 0,
):
    """Leiden clusters as proxy labels (paper: manual annotation of Leiden clusters)."""
    import scanpy as sc

    proteins = [
        v
        for v in adata.var_names
        if not any(tok in str(v) for tok in ("FSC", "SSC", "Time", "LD"))
    ]
    a = adata[:, proteins].copy()
    a.X = np.asarray(a.layers[layer])
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    # scanpy>=1.12 forwards `seed=` into igraph's community_leiden, which igraph 1.0.0 rejects
    # ("unexpected keyword argument"). Seed igraph's own RNG instead (it wants a random.Random,
    # not a numpy Generator) and drop the seed kwarg. n_iterations=2 matches scanpy's igraph-flavor
    # default; this keeps recompute deterministic and available at any resolution.
    import random

    import igraph

    igraph.set_random_number_generator(random.Random(seed))
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", directed=False, n_iterations=2)
    adata.obs[labels_key] = a.obs["leiden"].astype(str).values
    return adata


def _subsample_batches(adata, batch_key: str, max_cells: int, seed: int = 0):
    """Stratified subsample (paper uses 100k total; scib uses 10k per batch)."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    batch = np.asarray(adata.obs[batch_key].astype(str))
    n_batch = len(np.unique(batch))
    per = max(1, max_cells // n_batch)
    mask = np.zeros(len(batch), dtype=bool)
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        if len(idx) > per:
            idx = rng.choice(idx, size=per, replace=False)
        mask[idx] = True
    idx = np.where(mask)[0]

    # scipy 1.17 + numpy 2.x regression: anndata's copy-of-view calls
    # _subset_sparse which passes a (row_array, col_array) 2-tuple to scipy,
    # triggering _get_arrayXarray → csr_sample_values, which rejects int64 scalars.
    # Fix: bypass anndata's copy-of-view entirely. Use 1-D row indexing on the
    # parent CSR (mat.tocsr()[idx]) which goes through _get_submatrix →
    # csr_row_index (no int64 issue), then densify the tiny cytometry matrices.
    def _row_subset(mat, idx):
        if sp.issparse(mat):
            return np.asarray(mat.tocsr()[idx].todense())
        return mat[idx]

    return ad.AnnData(
        X=_row_subset(adata.X, idx),
        obs=adata.obs.iloc[idx].copy(),
        var=adata.var.copy(),
        layers={k: _row_subset(v, idx) for k, v in adata.layers.items()} or None,
        obsm={k: v[idx] for k, v in adata.obsm.items()} or None,
    )


def load_nunez(
    data_dir: str,
    auto_download: bool = True,
    *,
    max_cells: int | None = 100_000,
    labels_key: str = "cell_type",
    leiden_resolution: float = 0.05,
    annotate: bool = True,
    seed: int = 0,
    annotated_h5ad: str | None = "nunez_annotated.h5ad",
):
    """Load D2: Nuñez PBMC vignette.

    If ``nunez_annotated.h5ad`` exists (CytoVI tutorial manual labels), load it and skip
    FCS I/O plus on-the-fly Leiden. Otherwise read FCS, preprocess, merge, and optionally
    assign proxy Leiden labels (not interchangeable with tutorial names).
    """
    if annotated_h5ad:
        h5ad_path = _resolve_file(data_dir, annotated_h5ad)
        if os.path.exists(h5ad_path) and os.path.getsize(h5ad_path) > 0:
            import scanpy as sc

            merged = sc.read_h5ad(h5ad_path)
            if (
                labels_key != "cell_type"
                and labels_key not in merged.obs
                and "cell_type" in merged.obs
            ):
                merged.obs[labels_key] = merged.obs["cell_type"]
            if max_cells is not None and merged.n_obs > max_cells:
                merged = _subsample_batches(merged, "batch", max_cells, seed=seed)
            return merged

    from scvi.external import cytovi

    paths = {}
    for name in ("Nunez_PBMCs_batch1.fcs", "Nunez_PBMCs_batch2.fcs"):
        p = _resolve_file(data_dir, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            if not auto_download:
                raise FileNotFoundError(f"{p} missing; fetch it (README).")
            p = download(name, data_dir)
        paths[name] = p

    try:
        b1 = cytovi.read_fcs(paths["Nunez_PBMCs_batch1.fcs"], remove_markers=["Time", "LD", "-"])
        b2 = cytovi.read_fcs(paths["Nunez_PBMCs_batch2.fcs"], remove_markers=["Time", "LD", "-"])
    except ImportError as e:
        raise ImportError(
            "Reading .fcs needs an FCS reader (e.g. `readfcs`/`flowio`) installed in the env. "
            "Install it, or skip D2 and use D1 (Roider .h5ad needs no FCS reader)."
        ) from e
    for b in (b1, b2):
        cytovi.transform_arcsinh(b)
        cytovi.scale(b)
    merged = cytovi.merge_batches([b1, b2])
    merged.obs_names_make_unique()
    if annotate:
        _leiden_labels(merged, labels_key=labels_key, resolution=leiden_resolution, seed=seed)
    if max_cells is not None and merged.n_obs > max_cells:
        merged = _subsample_batches(merged, "batch", max_cells, seed=seed)
    return merged


def _subset_roider_patients(merged, max_patients: int | None):
    if max_patients is None:
        return merged
    keep = sorted(merged.obs["PatientID"].astype(str).unique())[:max_patients]
    return merged[merged.obs["PatientID"].astype(str).isin(keep)].copy()


def _split_by_entity(
    adata,
    ref_entities: list[str],
    query_entities: list[str],
    *,
    entity_key: str = "Entity",
):
    """Split ``adata`` into reference and query subsets by entity label.

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[entity_key]`` populated (e.g. by
        :func:`benchmarks.common.roider_metadata.annotate_roider_obs`).
    ref_entities
        Entity labels that belong to the reference atlas (e.g. ``["FL", "DLBCL"]``).
    query_entities
        Entity labels that belong to the novel query (e.g. ``["MCL", "rLN"]``).
    entity_key
        Column in ``adata.obs`` containing entity labels. Defaults to ``"Entity"`` (Roider
        convention).

    Returns
    -------
    ref_adata, query_adata : tuple of AnnData copies
        Cells with entity in ``ref_entities`` and ``query_entities`` respectively.
        Cells whose entity is not in either list are silently excluded.
    """
    entity = adata.obs[entity_key].astype(str)
    ref = adata[entity.isin(ref_entities)].copy()
    query = adata[entity.isin(query_entities)].copy()
    return ref, query


def load_roider_full(
    data_dir: str | None = None,
    *,
    raw_root: str | None = None,
    max_patients: int | None = None,
    max_cells_per_patient: int = 10_000,
    cache_path: str | None = None,
    seed: int = 0,
    annotate_metadata: bool = True,
    leiden_labels: bool = True,
    leiden_resolution: float = 1.0,
    labels_key: str = "cell_type",
    leiden_refresh: bool = False,
):
    """Load full Roider cohort from extracted ``data/roider_raw/`` FCS (issue cytovi-benchmark/02).

    Expects ``data/24915633.zip`` extracted via
    ``python -m benchmarks.common.ingest --extract-roider``.
    Panel-1 ``cell_type`` labels are Leiden clusters (proxy; not manual gating).
    """
    from pathlib import Path

    import anndata as ad
    import scanpy as sc

    from benchmarks.common.ingest import ROIDER_FCS, inventory_roider
    from scvi.external import cytovi

    repo_data = Path(_REPO_DATA)
    raw = Path(raw_root) if raw_root else repo_data / "roider_raw"
    cache = Path(cache_path) if cache_path else repo_data / "roider_full" / "merged.h5ad"
    if cache.exists() and cache.stat().st_size > 0:
        merged = _ensure_scaled(sc.read(cache))
        merged = _subset_roider_patients(merged, max_patients)
        p1 = merged[merged.obs["panel_batch"].astype(str) == "0"].copy()
        p2 = merged[merged.obs["panel_batch"].astype(str) == "1"].copy()
        if annotate_metadata:
            from benchmarks.common.roider_metadata import (
                annotate_roider_obs,
                load_patient_entity_map,
            )

            annotate_roider_obs(
                merged,
                p1,
                p2,
                load_patient_entity_map(),
                labels_key=labels_key,
                leiden=leiden_labels,
                leiden_resolution=leiden_resolution,
                leiden_refresh=leiden_refresh,
                leiden_write_cache=max_patients is None,
                seed=seed,
            )
        return merged, p1, p2

    if not raw.exists():
        raise FileNotFoundError(
            f"Roider raw tree missing at {raw}. Run: "
            "python -m benchmarks.common.ingest --extract-roider"
        )

    inventory_roider(raw)
    by_patient: dict[str, dict[str, Path]] = {}
    for path in raw.rglob("*.fcs"):
        if "Compensation" in path.name or "__MACOSX" in str(path):
            continue
        m = ROIDER_FCS.search(path.name)
        if not m:
            continue
        pid = m.group("sample")
        key = f"panel{m.group('panel')}"
        by_patient.setdefault(pid, {})[key] = path

    patients = sorted(by_patient)
    if max_patients is not None:
        patients = patients[:max_patients]
    if not patients:
        raise RuntimeError(f"No Panel1/Panel2 FCS found under {raw}")

    rng = np.random.default_rng(seed)
    panel1_list, panel2_list = [], []
    for pid in patients:
        paths = by_patient[pid]
        if "panel1" not in paths or "panel2" not in paths:
            continue
        for panel_key, adata_list in (("panel1", panel1_list), ("panel2", panel2_list)):
            a = cytovi.read_fcs(str(paths[panel_key]), remove_markers=["Time", "LD", "-"])
            cytovi.transform_arcsinh(a, global_scaling_factor=500)
            cytovi.scale(a)
            a.obs["PatientID"] = pid
            if a.n_obs > max_cells_per_patient:
                idx = rng.choice(a.n_obs, size=max_cells_per_patient, replace=False)
                a = a[idx].copy()
            adata_list.append(a)

    if not panel1_list or not panel2_list:
        raise RuntimeError("No paired panel1/panel2 patients loaded")

    # Inner join: patients can differ in exported marker sets; outer concat leaves NaNs
    # in ``scaled`` and ``merge_batches`` rejects those.
    p1 = ad.concat(panel1_list, join="inner", label="patient_batch")
    p2 = ad.concat(panel2_list, join="inner", label="patient_batch")
    p1.obs_names_make_unique()
    p2.obs_names_make_unique()
    merged = cytovi.merge_batches([p1, p2], batch_key="panel_batch")
    cache.parent.mkdir(parents=True, exist_ok=True)
    merged.write(cache)
    if annotate_metadata:
        from benchmarks.common.roider_metadata import annotate_roider_obs, load_patient_entity_map

        annotate_roider_obs(
            merged,
            p1,
            p2,
            load_patient_entity_map(),
            labels_key=labels_key,
            leiden=leiden_labels,
            leiden_resolution=leiden_resolution,
            leiden_refresh=leiden_refresh,
            leiden_write_cache=max_patients is None,
        )
    return merged, p1, p2


def make_synthetic_panels(seed: int = 0, n_genes: int = 30, backbone: int = 20):
    """Two synthetic panels sharing a backbone, for --smoke (no download). Mirrors the real shape.

    Returns ``(merged, p1, p2)`` with a 'scaled' layer, 'labels' (panel 1 only realistic, but kept
    on both for the holdout tasks), 'batch', and a '_nan_mask' for the panel-specific markers.
    """
    from scvi.data import synthetic_iid
    from scvi.external import cytovi

    def _one(s, ng):
        a = synthetic_iid(
            batch_size=256,
            n_genes=ng,
            n_proteins=0,
            n_regions=0,
            n_batches=2,
            n_labels=5,
            rna_dist="normal",
        )
        a.obs_names = [f"s{s}_{n}" for n in a.obs_names]
        a.layers["raw"] = a.X.copy()
        cytovi.transform_arcsinh(a)
        cytovi.scale(a)
        return a

    p1 = _one(seed, n_genes)  # full panel (backbone + panel-specific)
    p2 = _one(seed + 1, backbone)  # backbone-only panel (missing the panel-specific tail)
    merged = cytovi.merge_batches([p1, p2], batch_key="panel_batch")
    return merged, p1, p2
