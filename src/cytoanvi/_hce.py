from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hierarchical_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reachability_matrix: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hierarchical cross-entropy loss (Microsoft hce-classification).

    Port of the loss from https://github.com/microsoft/hce-classification (MIT).

    Parameters
    ----------
    logits
        Raw classifier outputs of shape ``(batch_size, n_labels)``.
    targets
        Ground-truth class indices of shape ``(batch_size,)``.
    reachability_matrix
        Binary reachability matrix ``R`` of shape ``(n_labels, n_labels)`` where
        ``R[i, j] = 1`` if class ``j`` is reachable from class ``i`` (``j`` is ``i`` or a
        descendant of ``i`` in the hierarchy).
    weight
        Optional per-class weights passed to :func:`~torch.nn.functional.nll_loss`.
    """
    probs = torch.softmax(logits, dim=-1)
    # Propagate probability mass from leaves to ancestors: hier_probs[b, i] = sum of
    # probs[b, j] for all j reachable from i (i.e. all descendants of i, including i itself).
    # .T is intentional — R[i, j]=1 means j is reachable FROM i; we want sum over j.
    hier_probs = torch.matmul(probs, reachability_matrix.T)
    eps = torch.finfo(hier_probs.dtype).eps
    log_probs = torch.log(hier_probs + eps)
    return F.nll_loss(log_probs, targets, weight=weight)


def validate_reachability_matrix(matrix: np.ndarray, n_labels: int) -> None:
    """Validate a reachability matrix for ``n_labels`` observed classes.

    Raises
    ------
    ValueError
        If shape or partial-order properties are violated.
    """
    arr = np.asarray(matrix)
    if arr.shape != (n_labels, n_labels):
        raise ValueError(
            f"reachability matrix must have shape ({n_labels}, {n_labels}); got {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("reachability matrix must be numeric.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("reachability matrix must be binary (entries 0 or 1).")
    if not np.all(np.diag(arr) == 1):
        raise ValueError("reachability matrix must be reflexive (diagonal entries are 1).")
    # Transitivity: if i reaches j and j reaches k, then i reaches k.
    reach = arr.astype(bool)
    closure = (reach @ reach) > 0  # bool: True where any shared intermediate j exists
    violations = closure & ~reach
    if violations.any():
        i, k = np.argwhere(violations)[0]
        # Recover a witness j: any j where reach[i,j] and reach[j,k].
        js = np.where(reach[i] & reach[:, k])[0]
        j = int(js[0])
        raise ValueError(
            "reachability matrix is not transitive "
            f"(class {j} reachable from {i}, {k} from {j}, but not {k} from {i})."
        )
    # Antisymmetry on distinct indices.
    off_diag = reach & reach.T & ~np.eye(n_labels, dtype=bool)
    if off_diag.any():
        i, j = np.argwhere(off_diag)[0]
        raise ValueError(
            "reachability matrix is not antisymmetric "
            f"(classes {i} and {j} are mutually reachable)."
        )


def build_reachability_matrix(label_names: list[str], edges: dict[str, list[str]]) -> np.ndarray:
    """Build a reachability matrix from a parent→children edge dictionary.

    Every name appearing in ``edges`` (as a parent key or a child value) must also appear
    in ``label_names``.  This means **all hierarchy nodes must be observed leaf labels** —
    virtual internal nodes that are not themselves predicted classes (e.g. a synthetic
    ``"root"`` aggregating several leaves) are not supported here.  If your hierarchy has
    such virtual nodes, pre-compute the reachability matrix manually and pass it directly
    via ``reachability_matrix`` to :meth:`~cytoanvi.CytoANVI.__init__` or
    :meth:`~cytoanvi.CytoANVI.set_hierarchy`.

    Internal nodes that *are* in ``label_names`` may be parents in ``edges``.  Every name
    in ``label_names`` must appear exactly once in the hierarchy graph (as a node) and must
    not appear as a child more than once.

    Parameters
    ----------
    label_names
        Observed class names in registry order (excluding the unlabeled category).
    edges
        Mapping from parent node to child nodes defining a DAG.

    Returns
    -------
    Binary array of shape ``(len(label_names), len(label_names))`` with
    ``R[i, j] = 1`` when class ``j`` is reachable from class ``i``.
    """
    if len(label_names) != len(set(label_names)):
        duplicates = sorted({n for n in label_names if label_names.count(n) > 1})
        raise ValueError(f"label_names must not contain duplicates; repeated names: {duplicates}.")

    if not edges and len(label_names) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if not edges and len(label_names) > 0:
        raise ValueError(
            "edges is empty but label_names is non-empty; provide a hierarchy or pass a "
            "reachability matrix directly."
        )

    children: dict[str, list[str]] = {}
    for parent, child_list in edges.items():
        if not isinstance(child_list, list):
            raise ValueError(
                f"edges[{parent!r}] must be a list of child names;"
                f" got {type(child_list).__name__}."
            )
        if parent in children:
            raise ValueError(f"duplicate parent key in edges: {parent!r}.")
        children[parent] = list(child_list)

    nodes: set[str] = set(children.keys())
    for child_list in children.values():
        nodes.update(child_list)

    missing = sorted(set(label_names) - nodes)
    if missing:
        raise ValueError(f"label_names not present in hierarchy edges: {missing}.")
    extra_nodes = sorted(nodes - set(label_names))
    if extra_nodes:
        raise ValueError(
            "hierarchy contains nodes not listed in label_names: "
            f"{extra_nodes}.  "
            "All nodes in the hierarchy (parents and children) must be observed leaf labels "
            "present in label_names — virtual internal nodes (e.g. a synthetic 'root' node) "
            "are not supported.  Either add the extra nodes to label_names, remove them from "
            "the hierarchy, or pre-compute the reachability matrix and pass it directly via "
            "reachability_matrix= instead of hierarchy_edges=."
        )

    child_counts: dict[str, int] = {}
    for child_list in children.values():
        for child in child_list:
            child_counts[child] = child_counts.get(child, 0) + 1
    duplicate_children = sorted(name for name, count in child_counts.items() if count > 1)
    if duplicate_children:
        raise ValueError(
            "each label must appear at most once as a child in the hierarchy; "
            f"duplicates: {duplicate_children}."
        )

    _validate_dag(children, nodes)

    name_to_idx = {name: idx for idx, name in enumerate(label_names)}
    n_labels = len(label_names)
    matrix = np.zeros((n_labels, n_labels), dtype=np.float32)

    def descendants(node: str) -> set[str]:
        result = {node}
        for child in children.get(node, []):
            result |= descendants(child)
        return result

    for i, name in enumerate(label_names):
        for desc in descendants(name):
            if desc in name_to_idx:
                matrix[i, name_to_idx[desc]] = 1.0

    validate_reachability_matrix(matrix, n_labels)
    return matrix


def _validate_dag(children: dict[str, list[str]], nodes: set[str]) -> None:
    """Raise ValueError if the parent→children graph contains a cycle."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_start = stack.index(node)
            cycle = stack[cycle_start:] + [node]
            raise ValueError("hierarchy edges contain a cycle: " + " -> ".join(cycle) + ".")
        visiting.add(node)
        stack.append(node)
        for child in children.get(node, []):
            if child not in nodes:
                raise ValueError(f"hierarchy edge references unknown node {child!r}.")
            dfs(child, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in nodes:
        dfs(node, [])
