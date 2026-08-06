"""Optional scHPL / treeArches helpers for CytoANVI latent hierarchies.

Requires ``pip install cytoanvi[cytoanvi-hierarchy]`` (scHPL). Import from
``cytoanvi.hierarchy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata
import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from cytoanvi._model import CytoANVI

_SCHPL_INSTALL_MSG = (
    "scHPL is required for hierarchy workflows. "
    "Install with: pip install cytoanvi[cytoanvi-hierarchy]"
)


def _require_schpl():
    try:
        import scHPL  # noqa: F401
    except ImportError as err:
        raise ImportError(_SCHPL_INSTALL_MSG) from err


def _get_tree_root(tree: Any):
    if isinstance(tree, list):
        if not tree:
            raise ValueError("scHPL tree is empty.")
        return tree[0]
    return tree


def _node_name(node: Any) -> str:
    name = node.name
    if isinstance(name, (list, tuple, np.ndarray)):
        parts = [str(x) for x in name]
        return parts[0] if len(parts) == 1 else " & ".join(parts)
    return str(name)


def _ensure_model_labels_represented(
    root: Any,
    model_labels: Sequence[str],
    leaf_to_model: dict[str, str],
) -> None:
    missing = []
    for label in model_labels:
        try:
            _representative_schpl_node(label, root, leaf_to_model)
        except ValueError:
            missing.append(label)
    if missing:
        raise ValueError(f"Model labels not represented in scHPL tree: {sorted(missing, key=str)}")


def _infer_leaf_to_model_mapping(
    schpl_leaves: Sequence[str], model_labels: Sequence[str]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for leaf in schpl_leaves:
        exact = [label for label in model_labels if leaf == label]
        prefixed = [label for label in model_labels if leaf.startswith(f"{label}-")]
        candidates = exact or prefixed
        if len(candidates) == 1:
            mapping[leaf] = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(f"Ambiguous mapping for scHPL leaf {leaf!r}: matches {candidates}")

    return mapping


def _is_ancestor(ancestor_node: Any, descendant_node: Any) -> bool:
    """Return True if ``ancestor_node`` is an ancestor of or equal to ``descendant_node``."""
    if ancestor_node is descendant_node:
        return True
    node = descendant_node.ancestor
    while node is not None:
        if node is ancestor_node:
            return True
        node = node.ancestor
    return False


def _iter_tree_nodes(root: Any):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.descendants))


def _representative_schpl_node(
    model_label: str,
    root: Any,
    leaf_to_model: dict[str, str],
) -> Any:
    """Map a CytoANVI observed label to a representative scHPL ``TreeNode``.

    Leaf nodes mapped via ``leaf_to_model`` are preferred. Internal nodes whose
    ``_node_name`` equals ``model_label`` are accepted when no leaf maps to it.
    """
    leaf_matches: list[Any] = []
    for leaf in root.get_leaves():
        schpl_leaf = _node_name(leaf)
        if leaf_to_model.get(schpl_leaf) == model_label:
            leaf_matches.append(leaf)
    if len(leaf_matches) > 1:
        raise ValueError(
            f"Ambiguous scHPL leaves for model label {model_label!r}: "
            f"{[_node_name(leaf) for leaf in leaf_matches]}"
        )
    if len(leaf_matches) == 1:
        return leaf_matches[0]

    internal_matches = [
        node
        for node in _iter_tree_nodes(root)
        if node.descendants and _node_name(node) == model_label
    ]
    if len(internal_matches) > 1:
        raise ValueError(
            f"Ambiguous internal scHPL nodes for model label {model_label!r}: "
            f"{[_node_name(node) for node in internal_matches]}"
        )
    if len(internal_matches) == 1:
        return internal_matches[0]

    raise ValueError(f"No scHPL node represents model label {model_label!r}.")


def reachability_from_schpl_tree(
    tree: Any,
    model_labels: Sequence[str],
    leaf_to_model: dict[str, str] | None = None,
) -> np.ndarray:
    """Convert an scHPL tree into an ``(n_labels, n_labels)`` reachability matrix."""
    from cytoanvi._hce import validate_reachability_matrix

    root = _get_tree_root(tree)
    label_names = list(model_labels)
    schpl_leaves = [_node_name(leaf) for leaf in root.get_leaves()]
    if leaf_to_model is None:
        leaf_to_model = _infer_leaf_to_model_mapping(schpl_leaves, label_names)

    _ensure_model_labels_represented(root, label_names, leaf_to_model)

    representatives = {
        label: _representative_schpl_node(label, root, leaf_to_model) for label in label_names
    }
    n_labels = len(label_names)
    matrix = np.zeros((n_labels, n_labels), dtype=np.float32)
    for i, ancestor_label in enumerate(label_names):
        ancestor_node = representatives[ancestor_label]
        for j, descendant_label in enumerate(label_names):
            descendant_node = representatives[descendant_label]
            if _is_ancestor(ancestor_node, descendant_node):
                matrix[i, j] = 1.0

    validate_reachability_matrix(matrix, n_labels)
    return matrix


def latent_to_anndata(
    model: CytoANVI,
    adata: AnnData,
    obs_cols: Sequence[str] | None = None,
) -> AnnData:
    """Wrap ``get_latent_representation`` as an AnnData matrix for scHPL / scanpy."""
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    latent = model.get_latent_representation(adata=adata)
    latent_adata = AnnData(X=latent)
    if obs_cols is not None:
        missing = [col for col in obs_cols if col not in adata.obs.columns]
        if missing:
            raise ValueError(f"obs columns not found in adata: {missing}")
        for col in obs_cols:
            latent_adata.obs[col] = adata.obs[col].values
    return latent_adata


def learn_hierarchy(
    latent_adata: AnnData,
    batch_key: str,
    batch_order: list,
    cell_type_key: str,
    **schpl_kwargs: Any,
):
    """Learn a cell-type hierarchy on aligned latents with scHPL ``learn_tree``."""
    _require_schpl()
    from scHPL.learn import learn_tree

    for col in (batch_key, cell_type_key):
        if col not in latent_adata.obs:
            raise ValueError(f"Column {col!r} not found in latent_adata.obs")

    return learn_tree(
        latent_adata,
        batch_key=batch_key,
        batch_order=batch_order,
        cell_type_key=cell_type_key,
        **schpl_kwargs,
    )


def update_hierarchy(
    latent_adata: AnnData,
    tree: Any,
    batch_added: list,
    batch_order: list,
    cell_type_key: str,
    batch_key: str,
    **kwargs: Any,
):
    """Update an existing scHPL tree with new batches (``retrain=False``)."""
    _require_schpl()
    if tree is None:
        raise ValueError("tree must not be None")
    from scHPL.learn import learn_tree

    for col in (batch_key, cell_type_key):
        if col not in latent_adata.obs:
            raise ValueError(f"Column {col!r} not found in latent_adata.obs")

    return learn_tree(
        latent_adata,
        batch_key=batch_key,
        batch_order=batch_order,
        cell_type_key=cell_type_key,
        tree=tree,
        retrain=False,
        batch_added=batch_added,
        **kwargs,
    )


def predict_schpl(query_latent: AnnData | np.ndarray, tree: Any, **kwargs: Any):
    """Predict hierarchical labels with scHPL ``predict_labels``."""
    _require_schpl()
    if tree is None:
        raise ValueError("tree must not be None")
    from scHPL.predict import predict_labels

    if isinstance(query_latent, AnnData):
        testdata = query_latent.X
    else:
        testdata = query_latent

    predictions, _probabilities = predict_labels(testdata, tree, **kwargs)
    return predictions


def set_hierarchy_from_schpl(
    model: CytoANVI,
    tree: Any,
    label_map: dict[str, str] | None = None,
) -> None:
    """Map an scHPL tree onto CytoANVI observed labels and call ``set_hierarchy``."""
    if tree is None:
        raise ValueError("tree must not be None")

    model_labels = model._observed_label_names()
    root = _get_tree_root(tree)
    schpl_leaves = [_node_name(leaf) for leaf in root.get_leaves()]

    if label_map is None:
        leaf_to_model = _infer_leaf_to_model_mapping(schpl_leaves, model_labels)
    else:
        leaf_to_model = dict(label_map)

    matrix = reachability_from_schpl_tree(tree, model_labels, leaf_to_model=leaf_to_model)
    model.set_hierarchy(matrix)


def _run_learn(
    reference_model: CytoANVI,
    reference_adata: AnnData | None,
    batch_key: str,
    batch_order: list,
    schpl_cell_type_key: str,
    tree: Any | None,
    obs_cols: Sequence[str] | None,
    **schpl_kwargs: Any,
) -> dict[str, Any]:
    """``mode='learn'`` branch: build the reference latent and call scHPL ``learn_tree``."""
    if reference_adata is None:
        raise ValueError("reference_adata is required for mode='learn'")
    if tree is not None:
        raise ValueError("tree must be None for mode='learn'")
    ref_latent = latent_to_anndata(reference_model, reference_adata, obs_cols=obs_cols)
    tree, missing = learn_hierarchy(
        ref_latent,
        batch_key=batch_key,
        batch_order=batch_order,
        cell_type_key=schpl_cell_type_key,
        **schpl_kwargs,
    )
    return {
        "tree": tree,
        "missing_populations": missing,
        "reference_latent": ref_latent,
    }


def _run_predict(
    query_latent: AnnData | np.ndarray | None,
    tree: Any,
    **schpl_kwargs: Any,
) -> dict[str, Any]:
    """``mode='predict'`` branch: run scHPL ``predict_labels`` against an existing tree."""
    if query_latent is None:
        raise ValueError("query_latent is required for mode='predict'")
    predictions = predict_schpl(query_latent, tree, **schpl_kwargs)
    return {"predictions": predictions, "tree": tree}


def _run_update(
    reference_adata: AnnData | None,
    query_adata: AnnData | None,
    combined_adata: AnnData | None,
    combined_latent: AnnData | None,
    query_model: CytoANVI | None,
    batch_added: list | None,
    batch_order: list,
    batch_key: str,
    schpl_cell_type_key: str,
    obs_cols: Sequence[str] | None,
    tree: Any,
    **schpl_kwargs: Any,
) -> dict[str, Any]:
    """``mode='update'`` branch: resolve the combined latent and call scHPL ``learn_tree``.

    Uses ``retrain=False``. Latent resolution (first match wins): ``combined_latent`` →
    ``combined_adata`` + ``query_model`` → ``anndata.concat([reference_adata, query_adata])``
    + ``query_model``.
    """
    if (
        query_adata is None
        and reference_adata is not None
        and combined_latent is None
        and combined_adata is None
    ):
        raise ValueError(
            "query_adata is required for mode='update' when reference_adata is provided "
            "without combined_latent or combined_adata."
        )
    if combined_latent is None and query_model is None:
        raise ValueError(
            "query_model is required for mode='update' when combined_latent is not provided "
            "(trained CytoANVI query model for latent extraction)."
        )
    if batch_added is None:
        raise ValueError("batch_added is required for mode='update'")

    if combined_latent is not None:
        latent_for_update = combined_latent
    elif combined_adata is not None:
        latent_for_update = latent_to_anndata(query_model, combined_adata, obs_cols=obs_cols)
    elif reference_adata is not None and query_adata is not None:
        merged_adata = anndata.concat(
            [reference_adata, query_adata], join="outer", index_unique="-"
        )
        latent_for_update = latent_to_anndata(query_model, merged_adata, obs_cols=obs_cols)
    else:
        raise ValueError(
            "mode='update' requires combined_latent, combined_adata, or both "
            "reference_adata and query_adata to build reference+query latents."
        )
    tree, missing = update_hierarchy(
        latent_for_update,
        tree=tree,
        batch_added=batch_added,
        batch_order=batch_order,
        cell_type_key=schpl_cell_type_key,
        batch_key=batch_key,
        **schpl_kwargs,
    )
    return {
        "tree": tree,
        "missing_populations": missing,
        "combined_latent": latent_for_update,
    }


def run_tree_arches_pipeline(
    reference_model: CytoANVI,
    batch_key: str,
    batch_order: list,
    cell_type_key: str,
    reference_adata: AnnData | None = None,
    query_adata: AnnData | None = None,
    query_latent: AnnData | None = None,
    combined_adata: AnnData | None = None,
    combined_latent: AnnData | None = None,
    tree: Any | None = None,
    mode: str = "learn",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fail-fast orchestrator chaining scHPL learn / update / predict on CytoANVI latents.

    Does not reimplement CytoANVI training — pass an already-trained reference (and, for
    ``mode='update'``, a trained query model via ``query_model`` in ``kwargs`` unless you pass
    pre-built ``combined_latent``).

    Update-mode latent resolution (first match wins): ``combined_latent`` → ``combined_adata`` +
    ``query_model`` → ``anndata.concat([reference_adata, query_adata])`` + ``query_model``.
    Query-only updates require ``combined_latent`` supplied by the caller.

    Thin dispatcher: each mode's logic lives in a private helper (``_run_learn``,
    ``_run_update``, ``_run_predict``) that takes explicit named parameters; this function
    only picks out the mode-specific keys from ``kwargs`` and routes to the right helper.
    """
    _require_schpl()
    reference_model._check_if_trained(warn=False)

    if mode not in ("learn", "update", "predict"):
        raise ValueError("mode must be one of 'learn', 'update', or 'predict'")

    if mode == "learn":
        obs_cols = kwargs.pop("obs_cols", None)
        schpl_cell_type_key = kwargs.pop("schpl_cell_type_key", cell_type_key)
        return _run_learn(
            reference_model=reference_model,
            reference_adata=reference_adata,
            batch_key=batch_key,
            batch_order=batch_order,
            schpl_cell_type_key=schpl_cell_type_key,
            tree=tree,
            obs_cols=obs_cols,
            **kwargs,
        )

    if tree is None:
        raise ValueError("tree is required for mode='update' and mode='predict'")

    if mode == "predict":
        return _run_predict(query_latent=query_latent, tree=tree, **kwargs)

    query_model = kwargs.pop("query_model", None)
    batch_added = kwargs.pop("batch_added", None)
    obs_cols = kwargs.pop("obs_cols", None)
    schpl_cell_type_key = kwargs.pop("schpl_cell_type_key", cell_type_key)
    return _run_update(
        reference_adata=reference_adata,
        query_adata=query_adata,
        combined_adata=combined_adata,
        combined_latent=combined_latent,
        query_model=query_model,
        batch_added=batch_added,
        batch_order=batch_order,
        batch_key=batch_key,
        schpl_cell_type_key=schpl_cell_type_key,
        obs_cols=obs_cols,
        tree=tree,
        **kwargs,
    )
