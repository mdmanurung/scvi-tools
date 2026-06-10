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

FIGSHARE = {
    "roider_p1.h5ad": "56891468",
    "roider_p2.h5ad": "56891471",
    "Nunez_PBMCs_batch1.fcs": "55982654",
    "Nunez_PBMCs_batch2.fcs": "55982657",
}
SCALED_LAYER = "scaled"
NAN_LAYER = "_nan_mask"


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
    for name in ("roider_p1.h5ad", "roider_p2.h5ad"):
        p = os.path.join(data_dir, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            if not auto_download:
                raise FileNotFoundError(
                    f"{p} missing; run with auto-download or fetch it (README)."
                )
            p = download(name, data_dir)
        paths[name] = p

    p1 = _ensure_scaled(sc.read(paths["roider_p1.h5ad"]))
    p2 = _ensure_scaled(sc.read(paths["roider_p2.h5ad"]))
    merged = cytovi.merge_batches([p1, p2], batch_key="panel_batch")
    return merged, p1, p2


def load_nunez(data_dir: str, auto_download: bool = True):
    """Load D2: read the two Nuñez FCS batches, arcsinh + scale + merge. Needs an FCS reader."""
    from scvi.external import cytovi

    paths = {}
    for name in ("Nunez_PBMCs_batch1.fcs", "Nunez_PBMCs_batch2.fcs"):
        p = os.path.join(data_dir, name)
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
    return merged


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
