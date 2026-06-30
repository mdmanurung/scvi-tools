"""Optional mapQC helpers for CytoANVI query-to-reference mapping QC.

Requires ``pip install scvi-tools[cytoanvi-mapping-qc]`` (mapqc). Import from
``cytoanvi.mapping_qc``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anndata import concat

if TYPE_CHECKING:
    from typing import Any, Literal

    from anndata import AnnData

    from cytoanvi._model import CytoANVI

_MAPQC_INSTALL_MSG = (
    "mapqc is required for mapping QC. "
    "Install with: pip install scvi-tools[cytoanvi-mapping-qc]"
)

DEFAULT_EMB_KEY = "X_CytoANVI"
DEFAULT_REF_Q_KEY = "mapqc_ref_query"
REF_CAT = "reference"
QUERY_CAT = "query"


def _require_mapqc() -> None:
    try:
        import mapqc  # noqa: F401
    except ImportError as err:
        raise ImportError(_MAPQC_INSTALL_MSG) from err


def build_mapqc_anndata(
    model: CytoANVI,
    reference_adata: AnnData,
    query_adata: AnnData,
    *,
    sample_key: str,
    emb_key: str = DEFAULT_EMB_KEY,
    ref_q_key: str = DEFAULT_REF_Q_KEY,
    r_cat: str = REF_CAT,
    q_cat: str = QUERY_CAT,
) -> AnnData:
    """Concatenate reference + query with a shared CytoANVI latent embedding for mapQC.

    The reference should contain **control / healthy cells only** (mapQC assumption). The query
    must include matched control cells (and optionally case cells) for :func:`evaluate_mapqc`.

    ``model`` should be a CytoANVI model that can embed **both** adatas (typically the trained
    query model after scArches surgery, not the reference-only model when query batches differ).
    """
    if sample_key not in reference_adata.obs:
        raise ValueError(f"sample_key {sample_key!r} not found in reference_adata.obs.")
    if sample_key not in query_adata.obs:
        raise ValueError(f"sample_key {sample_key!r} not found in query_adata.obs.")

    n_ref_samples = reference_adata.obs[sample_key].astype(str).nunique()
    if n_ref_samples < 3:
        raise ValueError(
            f"mapQC requires at least 3 reference samples in {sample_key!r}; got {n_ref_samples}."
        )
    if query_adata.n_obs == 0:
        raise ValueError("query_adata has no cells.")

    model._check_if_trained(warn=False)
    ref = reference_adata.copy()
    query = query_adata.copy()
    ref.obsm[emb_key] = model.get_latent_representation(ref)
    query.obsm[emb_key] = model.get_latent_representation(query)
    return concat([ref, query], label=ref_q_key, keys=[r_cat, q_cat], index_unique="-")


def run_mapqc_on_joint(
    joint_adata: AnnData,
    *,
    sample_key: str,
    adata_emb_loc: str = DEFAULT_EMB_KEY,
    ref_q_key: str = DEFAULT_REF_Q_KEY,
    q_cat: str = QUERY_CAT,
    r_cat: str = REF_CAT,
    n_nhoods: int,
    k_min: int,
    k_max: int,
    study_key: str | None = None,
    exclude_same_study: bool = False,
    grouping_key: str | None = None,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    seed: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> AnnData:
    """Run mapQC on a joint AnnData built by :func:`build_mapqc_anndata`."""
    _require_mapqc()
    from mapqc import run_mapqc

    run_mapqc(
        joint_adata,
        adata_emb_loc=adata_emb_loc,
        ref_q_key=ref_q_key,
        q_cat=q_cat,
        r_cat=r_cat,
        sample_key=sample_key,
        n_nhoods=n_nhoods,
        k_min=k_min,
        k_max=k_max,
        study_key=study_key,
        exclude_same_study=exclude_same_study,
        grouping_key=grouping_key,
        distance_metric=distance_metric,
        seed=seed,
        overwrite=overwrite,
        verbose=verbose,
        **kwargs,
    )
    return joint_adata


def run_mapqc_on_cytoanvi(
    model: CytoANVI,
    reference_adata: AnnData,
    query_adata: AnnData,
    *,
    sample_key: str,
    n_nhoods: int,
    k_min: int,
    k_max: int,
    emb_key: str = DEFAULT_EMB_KEY,
    ref_q_key: str = DEFAULT_REF_Q_KEY,
    q_cat: str = QUERY_CAT,
    r_cat: str = REF_CAT,
    study_key: str | None = None,
    exclude_same_study: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> AnnData:
    """Build a joint latent AnnData and run mapQC (requires the optional extra)."""
    joint = build_mapqc_anndata(
        model,
        reference_adata,
        query_adata,
        sample_key=sample_key,
        emb_key=emb_key,
        ref_q_key=ref_q_key,
        r_cat=r_cat,
        q_cat=q_cat,
    )
    return run_mapqc_on_joint(
        joint,
        sample_key=sample_key,
        adata_emb_loc=emb_key,
        ref_q_key=ref_q_key,
        q_cat=q_cat,
        r_cat=r_cat,
        n_nhoods=n_nhoods,
        k_min=k_min,
        k_max=k_max,
        study_key=study_key,
        exclude_same_study=exclude_same_study,
        verbose=verbose,
        **kwargs,
    )


def evaluate_mapqc(
    joint_adata: AnnData,
    case_control_key: str,
    case_cats: list[str],
    control_cats: list[str],
) -> dict:
    """Summarize mapQC output (requires prior :func:`run_mapqc_on_joint`)."""
    _require_mapqc()
    from mapqc import evaluate

    return evaluate(joint_adata, case_control_key, case_cats, control_cats)


def query_control_mapqc_rate(
    joint_adata: AnnData,
    *,
    control_value: str,
    case_control_key: str,
    ref_q_key: str = DEFAULT_REF_Q_KEY,
    q_cat: str = QUERY_CAT,
    score_threshold: float = 2.0,
) -> dict:
    """Fraction of query control cells with mapQC score above ``score_threshold``."""
    if "mapqc_score" not in joint_adata.obs:
        raise ValueError("mapqc_score not found in joint_adata.obs. Run mapQC first.")
    if case_control_key not in joint_adata.obs:
        raise ValueError(f"{case_control_key!r} not found in joint_adata.obs.")
    if ref_q_key not in joint_adata.obs:
        raise ValueError(f"{ref_q_key!r} not found in joint_adata.obs.")
    if "mapqc_filtering" not in joint_adata.obs:
        raise ValueError("mapqc_filtering not found in joint_adata.obs. Run mapQC first.")

    query = joint_adata.obs[joint_adata.obs[ref_q_key] == q_cat]
    controls = query[query[case_control_key].astype(str).isin([control_value])]
    passed = controls[controls["mapqc_filtering"] == "pass"]
    if len(passed) == 0:
        return {
            "n_query_controls": int(len(controls)),
            "n_pass": 0,
            "frac_dist_to_ref": float("nan"),
            "score_threshold": score_threshold,
        }
    distant = (passed["mapqc_score"] > score_threshold).mean()
    return {
        "n_query_controls": int(len(controls)),
        "n_pass": int(len(passed)),
        "frac_dist_to_ref": float(distant),
        "score_threshold": score_threshold,
    }
