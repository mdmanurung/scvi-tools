from __future__ import annotations

import contextlib
import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY, _SETUP_METHOD_NAME
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.external.cytovi import CYTOVI
from scvi.external.cytovi._constants import CYTOVI_REGISTRY_KEYS
from scvi.model.base import SemisupervisedTrainingMixin
from scvi.model.base._archesmixin import ArchesMixin, _get_loaded_data
from scvi.train._config import merge_kwargs
from scvi.utils import setup_anndata_dsp

from ._continual import ContinualUpdate, CytoANVIContinualTrainingPlan
from ._hce import build_reachability_matrix, validate_reachability_matrix
from ._module import _NON_DATA_INFERENCE_KEYS, CytoANVAE
from ._uncertainty import compute_uncertainty_scores

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class CytoANVI(SemisupervisedTrainingMixin, CYTOVI):
    """Semi-supervised, annotation-aware variational inference for cytometry (CytoANVI).

    CytoANVI extends :class:`~scvi.external.CYTOVI` the way :class:`~scvi.model.SCANVI` extends
    :class:`~scvi.model.SCVI`: it adds a cell-type classifier head on the shared latent space and
    a partially-observed-label objective, while preserving CytoVI's protein-specific likelihood,
    batch correction, missing-marker imputation, and scArches surgery. Labeled cells shape a more
    biologically meaningful latent space and labels are transferred to unlabeled cells.

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`~cytoanvi.CytoANVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space. If ``None``, set by a heuristic on input features.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    protein_likelihood
        Likelihood for protein expression: ``'normal'`` or ``'beta'``.
    latent_distribution
        Latent distribution: ``'normal'`` or ``'ln'``.
    encode_backbone_only
        If ``True``, only encode backbone markers (required for overlapping panels).
    encoder_marker_list
        Optional list of markers to use for encoding.
    linear_classifier
        If ``True``, uses a single linear layer for classification instead of an MLP.
    y_prior
        Prior over the observed labels. One of: ``"uniform"`` / ``None`` (uniform), ``"empirical"``
        (label frequencies among labeled cells, Laplace-smoothed), or a tensor of shape
        ``(1, n_labels)``. Use ``"empirical"`` for class-imbalanced panels.
    class_weighting
        Optional per-class weighting for the supervised classifier loss. ``"none"`` preserves
        default behavior. ``"inverse_frequency"`` and ``"sqrt_inverse_frequency"`` compute weights
        from labeled observed cells.
    class_weight_clip
        Maximum computed class weight before mean normalization.
    hierarchy_edges
        Optional parent→children edge dictionary defining a label hierarchy for hierarchical
        cross-entropy (HCE) training and :meth:`predict_hierarchical`. Mutually exclusive with
        ``reachability_matrix``. See :func:`~cytoanvi._hce.build_reachability_matrix`.
    reachability_matrix
        Optional precomputed binary reachability matrix of shape ``(n_labels, n_labels)``
        as a NumPy array or Tensor. ``R[i, j] = 1`` if label ``j`` is reachable from label
        ``i`` (i.e. a descendant-or-self). Mutually exclusive with ``hierarchy_edges``.
    **model_kwargs
        Keyword args for :class:`~cytoanvi.CytoANVAE`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> cytoanvi.CytoANVI.setup_anndata(
    ...     adata, batch_key="batch", labels_key="celltype", unlabeled_category="Unknown"
    ... )
    >>> model = cytoanvi.CytoANVI(adata)
    >>> model.train()
    >>> adata.obsm["X_CytoANVI"] = model.get_latent_representation()
    >>> adata.obs["pred"] = model.predict()

    Notes
    -----
    - Only ``latent_distribution="normal"`` is supported (the semi-supervised ``z1`` log-prob term
      assumes a Gaussian latent).
    - With overlapping panels, the encoder (and hence the classifier, which reads the shared
      latent ``z1``) only sees backbone markers. Cell types separated only by panel-specific
      markers may be under-resolved; consider adding discriminative markers to the backbone.
    - For query mapping via :meth:`load_query_data`, any labeled query cells must use labels
      already present in the reference; the classifier head is fixed at the reference ``n_labels``.
    - If the query was measured with a *different antibody panel* than the reference, call
      :meth:`prepare_query_anndata` first: it pads missing markers and masks them via CytoVI's
      ``nan_layer`` (rather than treating padded zeros as observed intensities). This requires the
      reference to have been set up with a ``nan_layer`` (i.e. a genuine backbone /
      panel-specific split). The query must fully observe the reference backbone; only
      panel-specific (non-backbone) markers may be absent.
    - :meth:`from_cytovi_model` returns an unfit model still in train mode; call
      ``model.module.eval()`` before :meth:`get_latent_representation` if inspecting it prior to
      :meth:`train`.
    - Optional hierarchical CE: :meth:`set_hierarchy` / :meth:`predict_hierarchical` (flat CE when
      no matrix is set). Optional scHPL treeArches helpers:
      ``cytoanvi.hierarchy`` (requires ``scvi-tools[cytoanvi-hierarchy]``).
    - Optional query-mapping QC: :meth:`score_query_mapping` wraps
      ``cytoanvi.mapping_qc`` (requires ``scvi-tools[cytoanvi-mapping-qc]``).
    """

    _module_cls = CytoANVAE

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int | None = None,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        protein_likelihood: Literal["normal", "beta"] = "normal",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_backbone_only: bool | None = None,
        encoder_marker_list: list | None = None,
        linear_classifier: bool = False,
        y_prior: Literal["uniform", "empirical"] | torch.Tensor | None = "uniform",
        class_weighting: Literal[
            "none", "inverse_frequency", "sqrt_inverse_frequency"
        ] | torch.Tensor | None = "none",
        class_weight_clip: float = 10.0,
        hierarchy_edges: dict[str, list[str]] | None = None,
        reachability_matrix: np.ndarray | torch.Tensor | None = None,
        **model_kwargs,
    ):
        if hierarchy_edges is not None and reachability_matrix is not None:
            raise ValueError(
                "Pass only one of hierarchy_edges or reachability_matrix, not both."
            )
        if latent_distribution != "normal":
            raise NotImplementedError(
                "CytoANVI only supports latent_distribution='normal'; the semi-supervised z1 "
                f"log-probability term assumes a Gaussian latent (got '{latent_distribution}')."
            )
        # Build the CytoVI backbone (validates beta range, resolves marker masking / n_latent
        # heuristic, registers backbone markers). The module it constructs is rebuilt below with
        # the label adjustment, mirroring TOTALANVI.from_totalvi/__init__.
        super().__init__(
            adata,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            encode_backbone_only=encode_backbone_only,
            encoder_marker_list=encoder_marker_list,
        )
        self._set_indices_and_labels()

        base = self.module
        # LabelsWithUnlabeledObsField appends the unlabeled category as the final code, so the
        # number of *observed* labels is one fewer than the registry's n_labels.
        n_labels = self.summary_stats.n_labels - 1
        if n_labels < 1:
            raise ValueError(
                "CytoANVI requires at least one observed label category (cells not equal to "
                f"unlabeled_category={self.unlabeled_category_!r}). All cells appear unlabeled."
            )

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(CYTOVI_REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if CYTOVI_REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        y_prior_tensor = self._resolve_y_prior(y_prior, n_labels)
        class_weights = self._resolve_class_weights(
            class_weighting, class_weight_clip, n_labels
        )
        self.class_weighting_ = (
            "tensor"
            if isinstance(class_weighting, torch.Tensor)
            else (class_weighting or "none")
        )
        self.class_weight_clip_ = (
            None if class_weights is None else float(class_weight_clip)
        )
        self.class_weights_ = (
            None if class_weights is None else class_weights.detach().cpu().numpy()
        )

        reachability_tensor = None
        if reachability_matrix is not None:
            reachability_arr = np.asarray(
                reachability_matrix.detach().cpu().numpy()
                if isinstance(reachability_matrix, torch.Tensor)
                else reachability_matrix
            )
            validate_reachability_matrix(reachability_arr, n_labels)
            reachability_tensor = torch.tensor(reachability_arr, dtype=torch.float32)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_labels=n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=base.n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            encoder_marker_mask=base.encoder_marker_mask,
            linear_classifier=linear_classifier,
            y_prior=y_prior_tensor,
            reachability_matrix=reachability_tensor,
            class_weights=class_weights,
            **model_kwargs,
        )

        if hierarchy_edges is not None:
            self.set_hierarchy(hierarchy_edges)
        elif reachability_tensor is not None:
            # Buffer already registered by CytoANVAE.__init__; only set the numpy mirror here.
            self.hierarchy_reachability_ = reachability_tensor.detach().cpu().numpy()

        self._model_summary_string = (
            f"CytoANVI Model with the following params: \nunlabeled_category: "
            f"{self.unlabeled_category_}, n_labels: {n_labels}, n_hidden: {n_hidden}, "
            f"n_latent: {base.n_latent}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"protein_likelihood: {protein_likelihood}, latent_distribution: {latent_distribution}"
        )
        self.unsupervised_history_ = None
        self.semisupervised_history_ = None
        self.was_pretrained = False
        self.n_labels = n_labels
        self._sync_encoder_marker_mask_attr()
        init_locals = locals()
        init_locals.pop("hierarchy_edges", None)
        init_locals.pop("reachability_matrix", None)
        init_locals.pop("class_weights", None)
        self.init_params_ = self._get_init_params(init_locals)

    @contextlib.contextmanager
    def _eval_mode(self):
        """Context manager: put ``self.module`` in eval mode and restore on exit."""
        was_training = self.module.training
        self.module.eval()
        try:
            yield
        finally:
            if was_training:
                self.module.train()

    def _sync_encoder_marker_mask_attr(self) -> None:
        """Persist backbone mask on the model for save/load and path-only query prep."""
        enc = getattr(self.module, "encoder_marker_mask", None)
        self.encoder_marker_mask_ = (
            np.asarray(enc, dtype=bool) if enc is not None else None
        )

    def _observed_label_names(self) -> list[str]:
        """Return observed label category names (excluding the unlabeled category)."""
        return list(self._label_mapping[: self.n_labels])

    def set_hierarchy(self, edges: dict[str, list[str]] | np.ndarray | torch.Tensor) -> None:
        """Set the label hierarchy used for HCE training and hierarchical prediction.

        Parameters
        ----------
        edges
            Either a parent→children edge dictionary (built with
            :func:`~cytoanvi._hce.build_reachability_matrix`) or a precomputed
            reachability matrix of shape ``(n_labels, n_labels)``.
        """
        label_names = self._observed_label_names()
        n_labels = len(label_names)
        if isinstance(edges, dict):
            edge_nodes: set[str] = set(edges.keys())
            for children in edges.values():
                edge_nodes.update(children)
            unknown = sorted(edge_nodes - set(label_names))
            if unknown:
                raise ValueError(
                    "hierarchy references labels not in the model's observed categories: "
                    f"{unknown}."
                )
            missing = sorted(set(label_names) - edge_nodes)
            if missing:
                raise ValueError(f"observed labels missing from hierarchy: {missing}.")
            matrix = build_reachability_matrix(label_names, edges)
        else:
            matrix = (
                edges.detach().cpu().numpy()
                if isinstance(edges, torch.Tensor)
                else np.asarray(edges)
            )
            validate_reachability_matrix(matrix, n_labels)
        tensor = torch.as_tensor(matrix, dtype=torch.float32)
        self.module._set_reachability(tensor)
        self.hierarchy_reachability_ = np.asarray(matrix, dtype=np.float32)

    @torch.inference_mode()
    def predict_hierarchical(
        self,
        adata: AnnData | None = None,
        soft: bool = False,
        leaf_only: bool = True,
        indices=None,
        batch_size: int | None = None,
        use_posterior_mean: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        """Predict cell types with scores propagated through the hierarchy.

        Parameters
        ----------
        adata
            AnnData registered with this model.
        soft
            If ``True``, return hierarchy-adjusted per-class scores. These scores are not
            normalized probabilities because ancestor scores include descendant mass.
        leaf_only
            If ``True`` (default) and ``soft=False``, restrict the argmax to leaf labels (no
            children in the hierarchy). With ``leaf_only=False`` the argmax runs over all nodes,
            and because ancestor scores include descendant mass it returns the highest-mass node
            (typically an internal/root node) -- intended only for inspecting subtree scores, not
            for leaf classification.
        indices
            Cell indices to predict.
        batch_size
            Minibatch size for inference.
        use_posterior_mean
            Whether to use the posterior mean of ``z`` for classification.

        Returns
        -------
        Hierarchical label predictions, or a DataFrame of hierarchy-adjusted scores if
        ``soft=True``.
        """
        if self.module.reachability_matrix_ is None:
            raise ValueError("No hierarchy set. Call set_hierarchy(...) first.")
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if len(indices) == 0:
            return (
                np.array([])
                if not soft
                else pd.DataFrame(
                    columns=self._observed_label_names(), index=adata.obs_names[indices]
                )
            )

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        y_pred = []
        with self._eval_mode():
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                data_inputs = {
                    key: inference_inputs[key]
                    for key in inference_inputs.keys()
                    if key not in _NON_DATA_INFERENCE_KEYS
                }
                batch = tensors[REGISTRY_KEYS.BATCH_KEY]
                cont_key = REGISTRY_KEYS.CONT_COVS_KEY
                cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
                cat_key = REGISTRY_KEYS.CAT_COVS_KEY
                cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

                pred = self.module.classify(
                    **data_inputs,
                    batch_index=batch,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    use_posterior_mean=use_posterior_mean,
                )
                if self.module.classifier.logits:
                    pred = torch.nn.functional.softmax(pred, dim=-1)
                pred = torch.matmul(pred, self.module.reachability_matrix_.T)
                if not soft:
                    if leaf_only:
                        leaf_mask = self.module.reachability_matrix_.sum(dim=-1) == 1
                        pred = pred.masked_fill(~leaf_mask, float("-inf")).argmax(dim=1)
                    else:
                        pred = pred.argmax(dim=1)
                y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred).numpy()
        if not soft:
            return np.array([self._code_to_label[p] for p in y_pred])
        return pd.DataFrame(
            y_pred,
            columns=self._observed_label_names(),
            index=adata.obs_names[indices],
        )

    def train(
        self,
        max_epochs: int | None = 1000,
        n_samples_per_label: float | None = None,
        lr: float = 1e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 4096,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
        adversarial_classifier: bool | None = None,
        plan_kwargs: dict | None = None,
        early_stopping_patience: int | None = 30,
        **kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. Default 1000.
        n_samples_per_label
            Number of subsamples for each label class to sample per epoch. By default, there
            is no label subsampling.
        lr
            Learning rate. Default 1e-3.
        accelerator
            Lightning accelerator. Default ``"auto"``.
        devices
            Devices to use. Default ``"auto"``.
        train_size
            Fraction of cells used for training. Default 0.9.
        validation_size
            Fraction used for validation. If ``None``, ``1 - train_size``.
        batch_size
            Minibatch size. Default 4096 — much larger than the scRNA-seq convention (128)
            because cytometry datasets are 100k–1M+ cells and large batches are required for
            gradient stability (small batches cause NaN divergence on large cohorts).
        early_stopping
            Stop when validation loss stops improving. Default ``True``.
        check_val_every_n_epoch
            How often to evaluate the validation set. Default: every epoch when
            ``early_stopping`` or ``reduce_lr_on_plateau`` is active.
        n_steps_kl_warmup
            Steps to linearly warm up the KL weight from 0 to 1. Active only when
            ``n_epochs_kl_warmup`` is ``None``.
        n_epochs_kl_warmup
            Epochs to warm up the KL weight. Default 400 (40% of ``max_epochs``).
        adversarial_classifier
            Use an adversarial classifier in the latent space for batch mixing. Defaults to
            ``True`` when missing-marker panels are detected.
        plan_kwargs
            Extra keyword arguments for the training plan (e.g.
            ``{"ewc_importance": 100}`` for continual update).
        early_stopping_patience
            Epochs to wait before triggering early stopping. Default 30.
        **kwargs
            Forwarded to ``Trainer``.
        """
        cont = getattr(self.module, "continual", None)
        if cont is not None and not cont.replay_batches:
            warnings.warn(
                "Continual update is active but the replay buffer is empty (typical after "
                "save/load). Experience replay is disabled until you re-call "
                "`load_query_data_with_replay(..., replay_adata=...)`. The EWC penalty still "
                "applies.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        plan_kwargs.update(update_dict)
        return super().train(
            max_epochs=max_epochs,
            n_samples_per_label=n_samples_per_label,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            adversarial_classifier=adversarial_classifier,
            plan_kwargs=plan_kwargs,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )

    @torch.inference_mode()
    def get_latent_representation(self, *args, **kwargs):
        """Compute latent representation with the module in eval mode."""
        with self._eval_mode():
            return super().get_latent_representation(*args, **kwargs)

    @classmethod
    def _encoder_mask_from_reference(
        cls, reference_model: str | CytoANVI, ref_var_names: pd.Index
    ) -> np.ndarray:
        """Resolve the reference backbone mask from an in-memory model or saved attrs."""
        if not isinstance(reference_model, str):
            enc = getattr(reference_model.module, "encoder_marker_mask", None)
            if enc is None or len(enc) != len(ref_var_names):
                raise ValueError(
                    "The reference has no usable encoder backbone mask (encoder_marker_mask); "
                    "panel-aware prep needs a reference whose nan_layer yields a genuine "
                    "backbone / panel-specific split."
                )
            return np.asarray(enc, dtype=bool)

        attr_dict, _, _, _ = _get_loaded_data(reference_model, device="cpu")
        enc = attr_dict.get("encoder_marker_mask_")
        if enc is None:
            raise ValueError(
                "Panel-aware prepare_query_anndata needs an in-memory reference model or a saved "
                "model that records encoder_marker_mask_ (re-save with scvi-tools >= 1.5). Load "
                "it first: reference_model = CytoANVI.load(path)."
            )
        backbone_mask = np.asarray(enc, dtype=bool)
        if len(backbone_mask) != len(ref_var_names):
            raise ValueError(
                "Saved encoder_marker_mask_ length does not match reference var names."
            )
        return backbone_mask

    @classmethod
    def select_replay_by_uncertainty(
        cls,
        model: CytoANVI,
        adata: AnnData,
        fraction: float = 0.2,
        tta_rep: int = 50,
    ) -> AnnData:
        """Select high-uncertainty reference cells for the continual replay buffer.

        Mirrors the cscanvi paper's Bregman-Information replay selection: cells whose latent
        embedding is unstable under feature masking are rehearsed during continual update.

        Parameters
        ----------
        model
            A trained :class:`CytoANVI` reference.
        adata
            Reference AnnData subset from which to draw the replay buffer.
        fraction
            Fraction of cells to retain (highest uncertainty first).
        tta_rep
            TTA repetitions passed to :meth:`get_uncertainty`.
        """
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")
        model._check_if_trained(warn=False)
        unc = model.get_uncertainty(adata, tta_rep=tta_rep)
        n = max(1, int(fraction * adata.n_obs))
        idx = np.argsort(unc)[-n:]
        return adata[idx].copy()

    def _resolve_y_prior(
        self, y_prior: Literal["uniform", "empirical"] | torch.Tensor | None, n_labels: int
    ) -> torch.Tensor | None:
        """Resolve ``y_prior`` into a ``(1, n_labels)`` tensor, or ``None`` for a uniform prior."""
        if y_prior is None or (isinstance(y_prior, str) and y_prior == "uniform"):
            return None
        if isinstance(y_prior, str):
            if y_prior != "empirical":
                raise ValueError(
                    f"y_prior must be 'uniform', 'empirical', None, or a tensor; got '{y_prior}'."
                )
            # frequencies of observed labels among labeled cells, Laplace-smoothed
            labeled_vals = self.labels_[self._labeled_indices]
            counts = np.array(
                [(labeled_vals == self._label_mapping[c]).sum() for c in range(n_labels)],
                dtype=np.float64,
            )
            counts += 1.0
            freqs = counts / counts.sum()
            return torch.tensor(freqs[None, :], dtype=torch.float32)
        if not isinstance(y_prior, torch.Tensor):
            raise ValueError(
                "y_prior must be 'uniform', 'empirical', None, or a tensor;"
                f" got {type(y_prior)!r}."
            )
        y_prior = y_prior.detach().clone().to(dtype=torch.float32)
        if tuple(y_prior.shape) != (1, n_labels):
            raise ValueError(
                f"y_prior tensor must have shape (1, {n_labels}); got {tuple(y_prior.shape)}."
            )
        if not torch.isfinite(y_prior).all():
            raise ValueError("y_prior tensor must contain only finite values.")
        if (y_prior < 0).any():
            raise ValueError("y_prior tensor must be non-negative.")
        row_sums = y_prior.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("y_prior tensor rows must sum to 1.")
        return y_prior

    def _resolve_class_weights(
        self,
        class_weighting: Literal[
            "none", "inverse_frequency", "sqrt_inverse_frequency"
        ] | torch.Tensor | None,
        class_weight_clip: float,
        n_labels: int,
    ) -> torch.Tensor | None:
        """Resolve optional classifier-loss weights into a 1-D tensor."""
        if class_weighting is None or (
            isinstance(class_weighting, str) and class_weighting == "none"
        ):
            return None
        if not np.isfinite(class_weight_clip) or class_weight_clip <= 0:
            raise ValueError("class_weight_clip must be finite and > 0.")
        if isinstance(class_weighting, torch.Tensor):
            weights = class_weighting.detach().clone().to(dtype=torch.float32)
        elif class_weighting in {"inverse_frequency", "sqrt_inverse_frequency"}:
            labeled_vals = self.labels_[self._labeled_indices]
            if len(labeled_vals) == 0:
                return None
            counts = np.array(
                [(labeled_vals == self._label_mapping[c]).sum() for c in range(n_labels)],
                dtype=np.float64,
            )
            if (counts <= 0).any():
                return None
            inv = counts.sum() / (n_labels * counts)
            weights_np = np.sqrt(inv) if class_weighting == "sqrt_inverse_frequency" else inv
            weights_np = np.minimum(weights_np, class_weight_clip)
            weights_np = weights_np / weights_np.mean()
            weights = torch.tensor(weights_np, dtype=torch.float32)
        else:
            raise ValueError(
                "class_weighting must be 'none', 'inverse_frequency', "
                "'sqrt_inverse_frequency', None, or a tensor."
            )
        if tuple(weights.shape) != (n_labels,):
            raise ValueError(
                f"class weights must have shape ({n_labels},); got {tuple(weights.shape)}."
            )
        if not torch.isfinite(weights).all():
            raise ValueError("class weights must contain only finite values.")
        if (weights <= 0).any():
            raise ValueError("class weights must be strictly positive.")
        return weights

    def _sync_class_weights_to_module(self) -> None:
        """Reattach persisted class weights to the non-persistent module buffer."""
        weights = getattr(self, "class_weights_", None)
        self.module.set_class_weights(
            None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        )

    @classmethod
    def from_cytovi_model(
        cls,
        cytovi_model: CYTOVI,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **cytoanvi_kwargs,
    ):
        """Initialize a CytoANVI model with weights from a pretrained CytoVI model.

        Parameters
        ----------
        cytovi_model
            Pretrained :class:`~scvi.external.CYTOVI` model.
        unlabeled_category
            Value used for unlabeled cells in ``labels_key``.
        labels_key
            Key in ``adata.obs`` for label information. If ``None``, uses the ``labels_key``
            used to set up the CytoVI model; if that was also ``None``, an error is raised.
        adata
            AnnData registered via :meth:`~cytoanvi.CytoANVI.setup_anndata`. If ``None``, uses
            the CytoVI model's AnnData.
        cytoanvi_kwargs
            Keyword args for the CytoANVI model.

        Returns
        -------
        CytoANVI
            A new, **untrained** :class:`CytoANVI` instance whose shared encoder and decoder
            weights are initialized from ``cytovi_model``. The classifier, ``encoder_z2_z1``,
            and ``decoder_z1_z2`` sub-networks are randomly initialized. Call :meth:`train`
            to fit the semi-supervised objective before using :meth:`predict` or
            :meth:`get_latent_representation`.
        """
        cytovi_model._check_if_trained(message="Passed in CytoVI model hasn't been trained yet.")

        cytoanvi_kwargs = dict(cytoanvi_kwargs)
        init_params = cytovi_model.init_params_
        non_kwargs = deepcopy(init_params["non_kwargs"])
        kwargs = deepcopy(init_params["kwargs"])
        # init_params_["kwargs"] groups params by category (e.g. {"encoder_kwargs": {...}}).
        # Flatten into a single dict; the group name is discarded.
        kwargs = {k: v for (_group, group_dict) in kwargs.items() for (k, v) in group_dict.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in cytoanvi_kwargs:
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"CytoVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del cytoanvi_kwargs[k]
        # `adata` is not a constructor kwarg to forward.
        non_kwargs.pop("adata", None)

        inherited_kwargs = {**non_kwargs, **kwargs}
        if inherited_kwargs.get("latent_distribution", "normal") != "normal":
            raise NotImplementedError(
                "CytoANVI.from_cytovi_model only supports inherited "
                "latent_distribution='normal'; train the CytoVI model with "
                "latent_distribution='normal' before warm-starting CytoANVI."
            )

        if adata is None:
            adata = cytovi_model.adata
        else:
            cytovi_model._validate_anndata(adata)

        cytovi_registry = cytovi_model.adata_manager.registry
        cytovi_setup_args = deepcopy(cytovi_registry[_SETUP_ARGS_KEY])
        cytovi_labels_key = cytovi_setup_args.get("labels_key")
        if cytovi_labels_key is None and labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the CytoVI model was initialized without one."
            )
        if cytovi_labels_key is not None:
            if labels_key is None:
                labels_key = cytovi_labels_key
            elif labels_key != cytovi_labels_key:
                raise ValueError(
                    "Cannot warm-start CytoANVI with a different labels_key than the "
                    f"pretrained CytoVI setup ({labels_key!r} != {cytovi_labels_key!r})."
                )
        cytovi_setup_args.update({"labels_key": labels_key})

        setup_method_name = cytovi_registry.get(_SETUP_METHOD_NAME, "setup_anndata")
        setup_method = getattr(cls, setup_method_name)
        setup_method(
            adata,
            unlabeled_category=unlabeled_category,
            **cytovi_setup_args,
        )

        new_model = cls(adata, **non_kwargs, **kwargs, **cytoanvi_kwargs)
        cytovi_state_dict = cytovi_model.module.state_dict()
        load_result = new_model.module.load_state_dict(cytovi_state_dict, strict=False)
        allowed_missing_prefixes = (
            "classifier.",
            "encoder_z2_z1.",
            "decoder_z1_z2.",
        )
        allowed_missing_keys = {"y_prior", "reachability_matrix_"}
        unexpected = set(load_result.unexpected_keys)
        disallowed_missing = {
            key
            for key in load_result.missing_keys
            if key not in allowed_missing_keys
            and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        }
        disallowed_unexpected = {
            key for key in unexpected if not key.startswith("prior_")
        }
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "Unexpected CytoVI -> CytoANVI state transfer mismatch. "
                f"Missing keys: {sorted(disallowed_missing)}; unexpected keys: "
                f"{sorted(disallowed_unexpected)}."
            )
        new_model_state_dict = new_model.module.state_dict()
        for key, cytovi_value in cytovi_state_dict.items():
            if key in unexpected:
                continue
            new_model_value = new_model_state_dict[key]
            if new_model_value.shape == cytovi_value.shape:
                torch.testing.assert_close(new_model_value.cpu(), cytovi_value.cpu())
        new_model.was_pretrained = True

        return new_model

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str,
        layer: str | None = None,
        batch_key: str | None = None,
        sample_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        nan_layer: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        %(param_unlabeled_category)s
        layer
            If not ``None``, uses this key in ``adata.layers`` for transformed protein expression.
        %(param_batch_key)s
        %(param_sample_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        nan_layer
            Optional layer key with a binary NaN feature mask for overlapping antibody panels.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(CYTOVI_REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(CYTOVI_REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(
                CYTOVI_REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
            ),
            CategoricalObsField(CYTOVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalJointObsField(
                CYTOVI_REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(CYTOVI_REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        if nan_layer is None and "_nan_mask" in adata.layers:
            msg = "Found nan_layer in adata. Registering nan_layer for missing-marker imputation."
            warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
            nan_layer = "_nan_mask"

        if nan_layer is not None:
            anndata_fields.append(LayerField(CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK, nan_layer))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def prepare_query_anndata(
        cls,
        adata: AnnData,
        reference_model: str | CytoANVI,
        return_reference_var_names: bool = False,
        inplace: bool = True,
    ) -> AnnData | pd.Index | None:
        """Panel-aware scArches query prep: pad missing markers **and** mask them.

        The gene-oriented :meth:`~scvi.model.base.ArchesMixin.prepare_query_anndata` pads markers
        absent from the query panel with **zeros**. For cytometry intensities zero is a real
        measurement, not "missing", so those padded markers would be read as observed-zero signal
        and corrupt both the embedding and the reconstruction loss. This override pads to the
        reference panel (reusing the base pad/sort) and additionally writes CytoVI's ``nan_layer``
        so the absent markers are **masked out** of the likelihood
        (``reconst_loss * nan_mask``) — mirroring how CYTOVI handles overlapping antibody panels
        (:func:`~scvi.external.cytovi.merge_batches` / ``register_nan_layer``). The padded,
        nan-masked query is then ready for :meth:`load_query_data`.

        Requires the reference to have been set up with a ``nan_layer`` so the masking field
        exists in the registry and :meth:`load_query_data` threads it through, **and** to have a
        genuine backbone / panel-specific split (an all-ones mask makes every marker a backbone
        marker, so no marker could ever be absent from the query — use a reference built from
        overlapping panels, e.g. via :func:`~scvi.external.cytovi.merge_batches`).

        Parameters
        ----------
        adata
            Query AnnData with its own (possibly smaller / reordered) antibody panel.
        reference_model
            A trained :class:`CytoANVI` reference (in memory or saved directory). Saved models
            must include ``encoder_marker_mask_`` (models saved with scvi-tools >= 1.5).
        return_reference_var_names
            If ``True``, only return the reference marker names (no padding/masking).
        inplace
            Whether to modify ``adata`` in place or return a new AnnData.

        Returns
        -------
        The padded, nan-masked query AnnData if ``inplace=False``; a :class:`pandas.Index` of
        reference markers if ``return_reference_var_names=True``; otherwise ``None``.
        """
        attr_dict, var_names, _, _ = _get_loaded_data(reference_model, device="cpu")
        ref_var_names = pd.Index(var_names)
        if return_reference_var_names:
            return ref_var_names
        backbone_mask = cls._encoder_mask_from_reference(reference_model, ref_var_names)

        setup_args = attr_dict["registry_"][_SETUP_ARGS_KEY]
        nan_layer_key = setup_args.get("nan_layer")
        if nan_layer_key is None:
            raise ValueError(
                "Panel-aware query prep requires the reference CytoANVI model to have been set "
                "up with a `nan_layer` (CytoANVI.setup_anndata(..., nan_layer=...)), so the "
                "registry has a masking field and a genuine backbone / panel-specific split. "
                "Without it, markers absent from the query panel cannot be masked out of the "
                "likelihood. A reference built from overlapping panels (e.g. via "
                "scvi.external.cytovi.merge_batches) registers this automatically. To instead "
                "treat missing markers as observed zeros, call "
                "scvi.model.base.ArchesMixin.prepare_query_anndata directly."
            )

        # Markers absent from the query panel (in reference order) — masked after padding.
        missing_markers = ref_var_names.difference(adata.var_names)

        backbone_markers = ref_var_names[backbone_mask]
        missing_backbone = backbone_markers.intersection(missing_markers)
        if len(missing_backbone):
            raise ValueError(
                f"Query panel is missing backbone (encoder) markers {list(missing_backbone)}. "
                "CytoVI encodes only the shared backbone, so the backbone must be present in "
                "both reference and query; only panel-specific (non-backbone) markers may be "
                "absent from the query. Add the missing backbone markers to the query, or use "
                "a reference whose backbone is shared with this query."
            )
        # A backbone marker the query *keeps* but masks in some cells (via its own pre-existing
        # nan_layer) drops out of the query-derived backbone just as a missing one would — but
        # `missing_markers` (set difference on var_names) can't see it. Catch it here, so it fails
        # fast with a clear message instead of a cryptic resize error in load_query_data.
        if nan_layer_key in adata.layers:
            qmask = np.asarray(adata.layers[nan_layer_key])
            present_backbone = backbone_markers.intersection(adata.var_names)
            if len(present_backbone):
                cols = adata.var_names.get_indexer(present_backbone)
                partial = present_backbone[(qmask[:, cols] == 0).any(axis=0)]
                if len(partial):
                    raise ValueError(
                        f"Query's nan_layer masks backbone (encoder) markers {list(partial)} in "
                        "some cells, so the query would re-derive a smaller backbone than the "
                        "reference and scArches surgery would fail. Backbone markers must be "
                        "fully observed across all query cells."
                    )
        # Panel-specific reference markers the query *did* measure: they'll be masked below so
        # the query re-derives the reference backbone, so their values won't be used. Warn.
        observed_nonbackbone = ref_var_names[~backbone_mask].intersection(adata.var_names)
        if len(observed_nonbackbone):
            warnings.warn(
                "Query measured reference panel-specific (non-backbone) markers "
                f"{list(observed_nonbackbone)}; these are masked so scArches re-derives the "
                "reference backbone, so their query values are not used for mapping.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        # Whether the query already carries a nan mask (overlapping internal panels). If so, the
        # base pad/sort already zeros the padded (missing) columns and preserves present-marker
        # mask values, so we only need to repair non-backbone columns below.
        query_had_mask = nan_layer_key in adata.layers

        # Base pad + sort: pads X and every existing layer with zeros, reorders to reference vars.
        result = ArchesMixin.prepare_query_anndata(adata, reference_model, inplace=inplace)
        target = adata if inplace else result

        if not query_had_mask:
            # 1 = observed, 0 = missing. All markers observed except the padded ones.
            mask = np.ones((target.n_obs, target.n_vars), dtype=np.float32)
            if len(missing_markers):
                mask[:, target.var_names.get_indexer(missing_markers)] = 0.0
            target.layers[nan_layer_key] = mask

        # Force every non-backbone reference marker to be masked, so the query re-derives exactly
        # the reference backbone (CytoVI's encoder reads only the backbone; an observed
        # panel-specific marker would otherwise enlarge the query backbone and break surgery).
        if not backbone_mask.all():
            mask = np.asarray(target.layers[nan_layer_key])
            mask[:, target.var_names.get_indexer(ref_var_names[~backbone_mask])] = 0.0
            target.layers[nan_layer_key] = mask

        return result

    @classmethod
    def load_query_data_with_replay(
        cls,
        adata: AnnData,
        reference_model: CytoANVI,
        replay_adata: AnnData,
        control_adata: AnnData | None = None,
        combine_type: str = "product",
        freeze_classifier: bool = True,
        seed: int = 0,
        **load_query_kwargs,
    ):
        """Continual case-control update: scArches surgery + Experience Replay + modified EWC.

        Implements the comparative-atlas (cscanvi) continual-learning update: maps ``adata`` onto
        ``reference_model`` (scArches), then
        trains with ``L = ELBO(query, replay) + (lambda/2)(F_reference o F_query_ctrl)
        (theta - theta_ref)^2`` — a replay buffer of reference cells is rehearsed in the ELBO, and
        the EWC penalty's Fisher weight is the Hadamard product of the reference-replay Fisher and
        the query-control Fisher. ``lambda`` is set at train time via
        ``train(plan_kwargs={"ewc_importance": ...})``.

        Parameters
        ----------
        adata
            Query AnnData (same vars as the reference; any labels must be reference labels).
        reference_model
            A trained :class:`CytoANVI` reference.
        replay_adata
            Replay buffer = a subset (~20%) of reference cells, rehearsed in the ELBO and used for
            the reference Fisher importances. Select randomly, or by :meth:`get_uncertainty` (BI).
        control_adata
            Healthy control cells from the query (~5-10%), used for the query-control Fisher.
            **Required** — controls must exist in both reference and query (the EWC term is
            ``F_reference o F_query_ctrl``).
        combine_type
            How to combine the two Fishers: ``"product"`` (paper default, Hadamard) or
            ``"additive"``.
        freeze_classifier
            Whether to freeze the classifier during surgery (passed to ``load_query_data``).
        seed
            RNG seed for the Fisher subsampling and replay-buffer ordering. Two calls with the
            same inputs and the same ``seed`` produce identical ``importances`` and replay batches.
        load_query_kwargs
            Additional keyword args for :meth:`~scvi.model.base.ArchesMixin.load_query_data`.

        Notes
        -----
        - ``ewc_importance`` (= lambda) is set at train time and is dataset-dependent (it scales
          against the Fisher magnitudes); tune it. ``0`` disables the EWC penalty (replay only).
          The paper used ``replay = 0.2`` (buffer fraction) and ``EWC = 100`` for scANVI/RNA;
          CytoVI's intensity likelihood has different Fisher magnitudes, so ``lambda`` must be
          retuned here rather than copied.
        - The continual update (:class:`~cytoanvi.ContinualUpdate`) is held by the
          module and persisted across :meth:`save` / :meth:`load` **except** its replay buffer
          (session-scoped). After a reload, ``predict`` / ``get_latent_representation`` /
          ``get_uncertainty`` work immediately; resuming continual *training* requires re-supplying
          ``replay_adata`` via another :meth:`load_query_data_with_replay`.
        - The query-control Fisher is computed on the batch-extended query model (controls may
          carry query batches); the reference anchor and reference Fisher on the reference model.
        """
        if combine_type not in ("additive", "product"):
            raise ValueError("combine_type must be 'additive' or 'product'.")
        if control_adata is None:
            raise ValueError(
                "control_adata is required: the paper's EWC term is F_reference o F_query_ctrl, "
                "so query control cells must be provided. Controls should exist in both reference "
                "and query."
            )

        model = cls.load_query_data(
            adata, reference_model, freeze_classifier=freeze_classifier, **load_query_kwargs
        )

        # One module owns the whole update (anchor, both Fishers, combine rule, replay buffer).
        model.module.continual = ContinualUpdate.configure(
            reference_model, model, replay_adata, control_adata,
            combine_type=combine_type, seed=seed,
        )
        # Persisted across save/load (replay buffer excluded); reattached in CytoANVAE.on_load.
        model.continual_update_state_ = model.module.continual.persistable_state()

        # route training through the EWC + experience-replay plan
        model._training_plan_cls = CytoANVIContinualTrainingPlan
        return model

    @classmethod
    def load_query_data(cls, *args, **kwargs):
        """Load query data and reattach non-persistent CytoANVI buffers."""
        model = super().load_query_data(*args, **kwargs)
        if hasattr(model, "_sync_class_weights_to_module"):
            model._sync_class_weights_to_module()
        return model

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata=None,
        accelerator: str = "auto",
        device: int | str = "auto",
        prefix: str | None = None,
        backup_url: str | None = None,
        datamodule=None,
        allowed_classes_names_list: list[str] | None = None,
    ):
        """Load a saved model; backfill ``encoder_marker_mask_`` from the module when absent."""
        if device == "cpu" and accelerator in {"auto", "cpu"}:
            accelerator = "cpu"
            device = "auto"
        model = super().load(
            dir_path,
            adata=adata,
            accelerator=accelerator,
            device=device,
            prefix=prefix,
            backup_url=backup_url,
            datamodule=datamodule,
            allowed_classes_names_list=allowed_classes_names_list,
        )
        if getattr(model, "encoder_marker_mask_", None) is None:
            model._sync_encoder_marker_mask_attr()
        if hasattr(model, "_sync_class_weights_to_module"):
            model._sync_class_weights_to_module()
        if getattr(model.module, "continual", None) is not None:
            model._training_plan_cls = CytoANVIContinualTrainingPlan
        return model

    def save(self, dir_path, prefix=None, overwrite=False, save_anndata=False, **kwargs):
        """Save model state, including ``encoder_marker_mask_`` for panel-aware query prep."""
        self._sync_encoder_marker_mask_attr()
        if getattr(self.module, "class_weights", None) is None:
            self.class_weights_ = None
        else:
            self.class_weights_ = self.module.class_weights.detach().cpu().numpy()
        if getattr(self.module, "reachability_matrix_", None) is not None:
            self.hierarchy_reachability_ = (
                self.module.reachability_matrix_.detach().cpu().numpy()
            )
        return super().save(
            dir_path,
            prefix=prefix,
            overwrite=overwrite,
            save_anndata=save_anndata,
            **kwargs,
        )

    @torch.inference_mode()
    def get_uncertainty(
        self,
        adata: AnnData | None = None,
        indices=None,
        batch_size: int | None = None,
        tta_rep: int = 50,
        mode: str = "latent",
    ) -> np.ndarray:
        """Per-cell Bregman-Information uncertainty via test-time augmentation.

        High scores flag cells whose embedding/logits are unstable under feature masking — a proxy
        for novelty / out-of-distribution query cells (e.g. disease-specific states absent from the
        reference). Useful before trusting :meth:`predict` on a mapped query.

        Parameters
        ----------
        tta_rep
            Number of TTA augmentations for estimating Bregman Information. More reps give a
            more stable estimate at linear cost. The default (50) balances stability and cost.
        mode
            ``"latent"`` (default): BI computed over encoder mean vectors (``n_latent`` dims).
            ``"logit"``: BI computed over classifier logit vectors (``n_labels`` dims) — the
            canonical Bregman-Variance-Decomposition-on-logits formulation.

        Returns
        -------
        np.ndarray of shape ``(n_cells,)`` with one non-negative scalar uncertainty score per
        cell (Bregman Information). Higher values indicate cells whose embedding or logits vary
        more across TTA augmentations — a proxy for novelty or out-of-distribution status.
        """
        if mode not in {"latent", "logit"}:
            raise ValueError("mode must be one of {'latent', 'logit'}.")
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        scores = []
        nan_key = CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK
        enc_mask = getattr(self.module, "encoder_marker_mask", None)
        with self._eval_mode():
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                nan_mask = None
                if nan_key in tensors:
                    full_mask = tensors[nan_key]
                    if enc_mask is not None:
                        nan_mask = full_mask[..., enc_mask]
                    else:
                        nan_mask = full_mask
                scores.append(
                    compute_uncertainty_scores(
                        inference_inputs, self.module, tta_rep=tta_rep,
                        nan_mask=nan_mask, mode=mode,
                    )
                )
        return torch.cat(scores).cpu().numpy()

    def score_query_mapping(
        self,
        reference_adata: AnnData,
        query_adata: AnnData,
        *,
        sample_key: str,
        n_nhoods: int,
        k_min: int,
        k_max: int,
        **kwargs,
    ) -> AnnData:
        """Run mapQC on CytoANVI latents after query-to-reference mapping.

        Requires ``pip install scvi-tools[cytoanvi-mapping-qc]``. Reference cells should be
        controls only; the query must include matched control cells.

        Parameters
        ----------
        reference_adata
            AnnData for reference (control) cells registered with this model.
        query_adata
            AnnData for query cells registered with this model.
        sample_key
            Key in ``adata.obs`` identifying biological samples (used to define neighborhoods
            for the mapQC scoring procedure).
        n_nhoods
            Number of neighborhoods to sample for mapQC scoring.
        k_min
            Minimum neighborhood size (number of nearest neighbors).
        k_max
            Maximum neighborhood size (number of nearest neighbors).
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~cytoanvi.mapping_qc.run_mapqc_on_cytoanvi`.

        Returns
        -------
        Joint AnnData combining reference and query cells, with ``mapqc_score`` written to
        ``obs`` for query cells only.
        """
        from cytoanvi import mapping_qc

        return mapping_qc.run_mapqc_on_cytoanvi(
            self,
            reference_adata,
            query_adata,
            sample_key=sample_key,
            n_nhoods=n_nhoods,
            k_min=k_min,
            k_max=k_max,
            **kwargs,
        )
