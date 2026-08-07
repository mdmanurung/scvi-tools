"""Fail-closed data and configuration contracts for MrTotalVI."""

from __future__ import annotations

import hashlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from anndata import AnnData


_VALID_U_PRIORS = frozenset({"standard", "mog", "vamp"})
_VALID_U_PRIOR_SUPERVISION = frozenset({"none", "labels"})


def resolve_u_prior(u_prior: str, legacy_mixture: bool | None) -> tuple[str, bool]:
    """Resolve the 0.2 prior enum and its narrow deprecated boolean migration."""
    if u_prior not in _VALID_U_PRIORS:
        raise ValueError("u_prior must be exactly one of {'standard', 'mog', 'vamp'}.")
    if legacy_mixture is not None and not isinstance(legacy_mixture, bool):
        raise TypeError("u_prior_mixture must be a bool or None.")

    allowed_legacy = {
        ("standard", False),
        ("mog", True),
        ("vamp", True),
    }
    if legacy_mixture is not None:
        if (u_prior, legacy_mixture) not in allowed_legacy:
            raise ValueError(
                "Contradictory u prior configuration: the only supported legacy "
                "combinations are ('standard', False), ('mog', True), and "
                "('vamp', True). Remove u_prior_mixture and use u_prior alone."
            )
        warnings.warn(
            "u_prior_mixture is deprecated and is accepted only as a checkpoint "
            "migration input; save again to persist the resolved u_prior enum.",
            DeprecationWarning,
            stacklevel=3,
        )
    return u_prior, u_prior != "standard"


def resolve_u_prior_supervision(
    supervision: str | None,
    weight: float,
    *,
    has_registered_labels: bool,
    legacy_checkpoint_hint: bool,
    resolved_prior: str,
) -> tuple[str, float]:
    """Resolve explicit supervision while retaining a narrow old-checkpoint migration."""
    try:
        resolved_weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise TypeError("u_prior_label_weight must be a finite number.") from exc
    if not np.isfinite(resolved_weight):
        raise ValueError("u_prior_label_weight must be finite.")

    if supervision is None:
        if resolved_weight == 0.0:
            supervision = "none"
        elif legacy_checkpoint_hint and resolved_weight > 0.0:
            # Historical label offsets existed only on the MoG path. Vamp
            # always used one global categorical distribution, and standard
            # has no mixture weights. Preserve that effective objective.
            if has_registered_labels and resolved_prior == "mog":
                supervision = "labels"
                warnings.warn(
                    "Missing u_prior_supervision metadata with a positive historical "
                    "u_prior_label_weight is migrated to explicit label supervision. "
                    "Save again to persist the resolved metadata.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            else:
                supervision = "none"
                resolved_weight = 0.0
                warnings.warn(
                    "Missing u_prior_supervision metadata in a historical checkpoint "
                    "is migrated to u_prior_supervision='none' and "
                    "u_prior_label_weight=0.0 because its historical effective "
                    "objective was unconditioned. Save again to persist the resolved metadata.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            raise ValueError(
                "A nonzero u_prior_label_weight requires explicit "
                "u_prior_supervision='labels'."
            )
    if supervision not in _VALID_U_PRIOR_SUPERVISION:
        raise ValueError("u_prior_supervision must be exactly one of {'none', 'labels'}.")

    if supervision == "none":
        if resolved_weight != 0.0:
            raise ValueError(
                "u_prior_supervision='none' requires u_prior_label_weight=0.0."
            )
        return "none", 0.0

    if not has_registered_labels:
        raise ValueError(
            "u_prior_supervision='labels' requires labels_key to be registered."
        )
    if resolved_weight <= 0.0:
        raise ValueError(
            "u_prior_supervision='labels' requires a finite positive "
            "u_prior_label_weight."
        )
    if resolved_prior == "standard":
        raise ValueError(
            "u_prior_supervision='labels' requires u_prior='mog' or 'vamp'; "
            "a standard Gaussian has no label-conditioned mixture weights."
        )
    return "labels", resolved_weight


def ordered_indices_sha256(indices: Sequence[int] | np.ndarray) -> str:
    """Hash an ordered integer index sequence with an unambiguous representation."""
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    header = f"int64:{values.size}:".encode()
    return hashlib.sha256(header + values.astype("<i8", copy=False).tobytes()).hexdigest()


def _as_dense_block(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        return np.asarray(value.toarray())
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    if isinstance(value, pd.Series):
        return value.to_numpy()
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


def _iter_matrix_values(matrix: Any, *, chunk_size: int = 1024):
    if sparse.issparse(matrix):
        yield np.asarray(matrix.data)
        return
    if isinstance(matrix, pd.DataFrame):
        for start in range(0, matrix.shape[0], chunk_size):
            yield matrix.iloc[start : start + chunk_size].to_numpy()
        return

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("Count input must be a two-dimensional matrix.")
    for start in range(0, int(shape[0]), chunk_size):
        block = matrix[start : start + chunk_size]
        yield _as_dense_block(block)


def validate_count_matrix(matrix: Any, *, name: str, chunk_size: int = 1024) -> None:
    """Exhaustively require finite, non-negative, integer-like count values."""
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional count matrix.")

    offset = 0
    for raw in _iter_matrix_values(matrix, chunk_size=chunk_size):
        values = np.asarray(raw).reshape(-1)
        if values.size == 0:
            continue
        if np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError(f"{name} must contain real-valued raw counts, not complex values.")
        if values.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(
                f"{name} must use a real numeric dtype for raw counts; "
                f"got {values.dtype}."
            )
        try:
            numeric = values.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric raw counts.") from exc

        finite = np.isfinite(numeric)
        if not finite.all():
            local = int(np.flatnonzero(~finite)[0])
            raise ValueError(
                f"{name} contains a non-finite value at scanned offset {offset + local}."
            )
        non_negative = numeric >= 0.0
        if not non_negative.all():
            local = int(np.flatnonzero(~non_negative)[0])
            raise ValueError(
                f"{name} contains a negative value at scanned offset {offset + local}."
            )
        integer_like = np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-8)
        if not integer_like.all():
            local = int(np.flatnonzero(~integer_like)[0])
            raise ValueError(
                f"{name} contains a non-integer-like value at scanned offset "
                f"{offset + local}."
            )
        offset += values.size


def validate_anndata_counts(
    adata: AnnData,
    *,
    layer: str | None,
    protein_expression_obsm_key: str,
) -> None:
    """Validate RNA and protein counts before setup mutates ``adata``."""
    if layer is None:
        rna = adata.X
        rna_name = "adata.X RNA"
    else:
        if layer not in adata.layers:
            raise KeyError(f"RNA count layer {layer!r} is absent from adata.layers.")
        rna = adata.layers[layer]
        rna_name = f"adata.layers[{layer!r}] RNA"
    if protein_expression_obsm_key not in adata.obsm:
        raise KeyError(
            f"Protein count matrix {protein_expression_obsm_key!r} is absent from adata.obsm."
        )
    validate_count_matrix(rna, name=rna_name)
    validate_count_matrix(
        adata.obsm[protein_expression_obsm_key],
        name=f"adata.obsm[{protein_expression_obsm_key!r}] protein",
    )


def take_matrix_rows(matrix: Any, indices: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return dense rows with DataFrame/sparse/backed parity and stable order."""
    rows = np.asarray(indices, dtype=np.int64).reshape(-1)
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[rows].to_numpy()
    if sparse.issparse(matrix):
        return matrix[rows].toarray()
    try:
        value = matrix[rows]
    except (TypeError, ValueError, IndexError):
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        sorted_value = _as_dense_block(matrix[sorted_rows])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return np.asarray(sorted_value)[inverse]
    return _as_dense_block(value)


def matrix_row(matrix: Any, index: int) -> np.ndarray:
    """Return one dense row without interpreting a DataFrame index as a column."""
    if isinstance(matrix, pd.DataFrame):
        value = matrix.iloc[int(index)].to_numpy()
    else:
        value = matrix[int(index)]
    return _as_dense_block(value).reshape(-1)


def authoritative_protein_names(
    adata: AnnData,
    *,
    protein_expression_obsm_key: str,
    protein_names_uns_key: str | None,
    required: bool,
) -> np.ndarray | None:
    """Resolve and validate an authoritative protein axis for one AnnData."""
    matrix = adata.obsm[protein_expression_obsm_key]
    width = int(matrix.shape[1])

    def _validated_names(values: Any, *, source: str) -> np.ndarray:
        names = np.asarray(values, dtype=object)
        if names.ndim != 1:
            raise ValueError(f"{source} must be a one-dimensional sequence of protein names.")
        if names.size == 0 or width == 0:
            raise ValueError("The authoritative protein axis must not be empty.")
        if names.size != width:
            raise ValueError(
                f"{source} has {names.size} entries but the matrix has {width} columns."
            )
        normalized: list[str] = []
        for value in names.tolist():
            # Check the scalar type before any missingness predicate: pd.isna on
            # a list-like value returns an array whose truth value is ambiguous.
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Protein names must be non-empty strings.")
            normalized.append(value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Protein names must be unique.")
        return np.asarray(normalized, dtype=object)

    dataframe_names = None
    if isinstance(matrix, pd.DataFrame):
        dataframe_names = _validated_names(
            matrix.columns,
            source=f"adata.obsm[{protein_expression_obsm_key!r}] DataFrame columns",
        )

    uns_names = None
    if protein_names_uns_key is not None:
        if protein_names_uns_key not in adata.uns:
            raise KeyError(
                f"Protein-name authority {protein_names_uns_key!r} is absent from adata.uns."
            )
        uns_names = _validated_names(
            adata.uns[protein_names_uns_key],
            source=f"adata.uns[{protein_names_uns_key!r}]",
        )

    if dataframe_names is not None and uns_names is not None:
        if not np.array_equal(dataframe_names, uns_names):
            raise ValueError(
                "Protein names in DataFrame columns and protein_names_uns_key must "
                "agree exactly in name and order."
            )
        return dataframe_names
    if dataframe_names is not None:
        return dataframe_names
    if uns_names is not None:
        return uns_names
    if required:
        raise ValueError(
            "Multi-file protein inputs require authoritative names from DataFrame "
            "columns or protein_names_uns_key; equal width is insufficient."
        )
    return None


def validate_sample_metadata(
    obs: pd.DataFrame,
    *,
    sample_key: str,
    covariate_keys: Sequence[str] = (),
    donor_key: str | None = None,
    sample_subset: Sequence[Any] | None = None,
    authoritative_order: Sequence[Any] | None = None,
) -> tuple[list[Any], pd.DataFrame]:
    """Validate sample mappings and return samples in the declared subset order."""
    requested_columns = [sample_key, *covariate_keys]
    if donor_key is not None:
        requested_columns.append(donor_key)
    requested_columns = list(dict.fromkeys(requested_columns))
    missing = [key for key in requested_columns if key not in obs.columns]
    if missing:
        raise KeyError(f"Missing required sample metadata columns: {missing}.")

    for key in requested_columns:
        missing_mask = obs[key].isna()
        if bool(missing_mask.any()):
            raise ValueError(
                f"Sample metadata column {key!r} contains null values at "
                f"{int(missing_mask.sum())} cells."
            )

    observed = list(pd.unique(obs[sample_key]))
    authority = list(authoritative_order) if authoritative_order is not None else observed
    if len(authority) != len(set(authority)):
        raise ValueError("The authoritative sample order contains duplicates.")
    unknown_observed = [value for value in observed if value not in set(authority)]
    missing_observed = [value for value in authority if value not in set(observed)]
    if unknown_observed or missing_observed:
        raise ValueError(
            "Observed samples do not match the authoritative registered sample order: "
            f"unknown={unknown_observed}, absent={missing_observed}."
        )

    if sample_subset is None:
        selected = authority
    else:
        if isinstance(sample_subset, (str, bytes)):
            raise TypeError("sample_subset must be a non-string sequence of sample names.")
        selected = list(sample_subset)
        if not selected:
            raise ValueError("sample_subset must not be empty.")
        if len(selected) != len(set(selected)):
            raise ValueError("sample_subset must not contain duplicate sample names.")
        unknown = [value for value in selected if value not in set(authority)]
        if unknown:
            raise ValueError(f"sample_subset contains unknown sample names: {unknown}.")

    mapping_keys = list(dict.fromkeys([*covariate_keys, *([donor_key] if donor_key else [])]))
    for key in mapping_keys:
        counts = obs.groupby(sample_key, observed=True, dropna=False)[key].nunique(dropna=False)
        bad = counts[counts != 1]
        if len(bad):
            raise ValueError(
                f"Sample metadata {key!r} is not constant within {sample_key!r}; "
                "each sample must map to exactly one non-null value. "
                f"invalid samples: {bad.index.tolist()}."
            )

    info = obs[requested_columns].drop_duplicates().set_index(sample_key)
    if info.index.has_duplicates:
        duplicated = info.index[info.index.duplicated()].unique().tolist()
        raise ValueError(f"Ambiguous sample metadata mappings for samples: {duplicated}.")
    return selected, info.loc[selected]
