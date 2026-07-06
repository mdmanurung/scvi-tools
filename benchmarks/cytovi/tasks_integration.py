"""A2 — batch integration benchmark (paper Figure 2E)."""

from __future__ import annotations

from benchmarks.common.baselines import cycombinepy_available, run_cycombinepy, run_harmony
from benchmarks.common.preprocessing import BENCHMARK_LAYER, PreprocScheme, apply_preproc_scheme
from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import latent_obsm, train_cytovi

PREPROC_SCHEMES: tuple[PreprocScheme, ...] = ("minmax", "zscore", "rank")


def _cytovi_integration(
    adata,
    *,
    batch_key: str,
    labels_key: str,
    scheme: PreprocScheme,
    source_layer: str = "scaled",
    max_epochs: int = 1000,
    n_latent: int | None = None,
    seed: int = 0,
    subsample_per_batch: int = 10_000,
):
    a = adata.copy()
    apply_preproc_scheme(a, scheme, source_layer=source_layer, out_layer=BENCHMARK_LAYER)
    model, a = train_cytovi(
        a,
        batch_key=batch_key,
        labels_key=labels_key,
        layer=BENCHMARK_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    latent_obsm(a, model, obsm_key=LATENT_OBSM)
    scib = run_scib_benchmark(
        a,
        batch_key=batch_key,
        label_key=labels_key,
        embedding_obsm_key=LATENT_OBSM,
        subsample_per_batch=subsample_per_batch,
        seed=seed,
    )
    return {
        "method": "cytovi",
        "preproc": scheme,
        "max_epochs": max_epochs,
        "n_latent": model.module.n_latent,
        **scib,
    }


def _expression_method_integration(
    adata,
    method: str,
    *,
    batch_key: str,
    labels_key: str,
    scheme: PreprocScheme,
    source_layer: str = "scaled",
    seed: int = 0,
    subsample_per_batch: int = 10_000,
):
    if method == "harmony":
        a = run_harmony(adata, batch_key=batch_key, scheme=scheme, source_layer=source_layer)
    elif method == "cycombinepy":
        if not cycombinepy_available():
            return {
                "method": "cycombinepy",
                "preproc": scheme,
                "skipped": True,
                "reason": "cycombinepy not installed",
            }
        a = run_cycombinepy(
            adata, batch_key=batch_key, scheme=scheme, source_layer=source_layer, seed=seed
        )
    else:
        raise ValueError(method)

    scib = run_scib_benchmark(
        a,
        batch_key=batch_key,
        label_key=labels_key,
        embedding_obsm_key=LATENT_OBSM,
        subsample_per_batch=subsample_per_batch,
        seed=seed,
    )
    return {"method": method, "preproc": scheme, **scib}


def task_a2_integration(
    adata,
    *,
    labels_key: str = "labels",
    batch_key: str = "batch",
    max_epochs: int = 1000,
    n_latent: int | None = None,
    seed: int = 0,
    subsample_per_batch: int = 10_000,
    schemes: tuple[PreprocScheme, ...] = PREPROC_SCHEMES,
    include_cycombinepy: bool = True,
    include_harmony: bool = True,
):
    """Run paper Figure 2E-style integration comparison with scib-metrics."""
    results = []
    for scheme in schemes:
        results.append(
            _cytovi_integration(
                adata,
                batch_key=batch_key,
                labels_key=labels_key,
                scheme=scheme,
                max_epochs=max_epochs,
                n_latent=n_latent,
                seed=seed,
                subsample_per_batch=subsample_per_batch,
            )
        )
        if include_harmony:
            results.append(
                _expression_method_integration(
                    adata,
                    "harmony",
                    batch_key=batch_key,
                    labels_key=labels_key,
                    scheme=scheme,
                    seed=seed,
                    subsample_per_batch=subsample_per_batch,
                )
            )
        if include_cycombinepy:
            results.append(
                _expression_method_integration(
                    adata,
                    "cycombinepy",
                    batch_key=batch_key,
                    labels_key=labels_key,
                    scheme=scheme,
                    seed=seed,
                    subsample_per_batch=subsample_per_batch,
                )
            )

    return {
        "task": "a2_integration",
        "seed": seed,
        "max_epochs": max_epochs,
        "subsample_per_batch": subsample_per_batch,
        "cycombinepy_note": (
            "cyCombinePy provides batch correction only (no marker imputation). "
            "See https://github.com/mdmanurung/cyCombinePy"
        ),
        "results": results,
    }
