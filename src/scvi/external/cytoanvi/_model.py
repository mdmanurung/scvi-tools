from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch

from scvi import settings
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
from scvi.utils import setup_anndata_dsp

from ._continual import (
    CytoANVIContinualTrainingPlan,
    compute_uncertainty_scores,
    zerolike_params_dict,
)
from ._module import CytoANVAE

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
        AnnData object registered via :meth:`~scvi.external.CytoANVI.setup_anndata`.
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
    **model_kwargs
        Keyword args for :class:`~scvi.external.cytoanvi.CytoANVAE`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.external.CytoANVI.setup_anndata(
    ...     adata, batch_key="batch", labels_key="celltype", unlabeled_category="Unknown"
    ... )
    >>> model = scvi.external.CytoANVI(adata)
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
    - :meth:`from_cytovi_model` returns an unfit model still in train mode; call
      ``model.module.eval()`` before :meth:`get_latent_representation` if inspecting it prior to
      :meth:`train`.
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
        y_prior: str | torch.Tensor | None = "uniform",
        **model_kwargs,
    ):
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

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(CYTOVI_REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if CYTOVI_REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        y_prior_tensor = self._resolve_y_prior(y_prior, n_labels)

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
            **model_kwargs,
        )

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
        self.init_params_ = self._get_init_params(locals())

    def _resolve_y_prior(
        self, y_prior: str | torch.Tensor | None, n_labels: int
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
        # assume a user-provided tensor of shape (1, n_labels)
        return y_prior

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
            AnnData registered via :meth:`~scvi.external.CytoANVI.setup_anndata`. If ``None``, uses
            the CytoVI model's AnnData.
        cytoanvi_kwargs
            Keyword args for the CytoANVI model.
        """
        cytovi_model._check_if_trained(message="Passed in CytoVI model hasn't been trained yet.")

        cytoanvi_kwargs = dict(cytoanvi_kwargs)
        init_params = cytovi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
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

        if adata is None:
            adata = cytovi_model.adata
        else:
            cytovi_model._validate_anndata(adata)

        cytovi_registry = cytovi_model.adata_manager.registry
        cytovi_setup_args = deepcopy(cytovi_registry[_SETUP_ARGS_KEY])
        cytovi_labels_key = cytovi_setup_args.get("labels_key")
        if labels_key is None and cytovi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the CytoVI model was initialized without one."
            )
        if labels_key is not None:
            cytovi_setup_args.update({"labels_key": labels_key})

        setup_method_name = cytovi_registry.get(_SETUP_METHOD_NAME, "setup_anndata")
        setup_method = getattr(cls, setup_method_name)
        setup_method(
            adata,
            unlabeled_category=unlabeled_category,
            **cytovi_setup_args,
        )

        cytoanvi_model = cls(adata, **non_kwargs, **kwargs, **cytoanvi_kwargs)
        cytovi_state_dict = cytovi_model.module.state_dict()
        cytoanvi_model.module.load_state_dict(cytovi_state_dict, strict=False)
        cytoanvi_model.was_pretrained = True

        return cytoanvi_model

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

    # ------------------------------------------------------------------ #
    # Continual / case-control atlas building (cscanvi-style EWC update)  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_importances(reference_model, adata) -> list[tuple[str, torch.Tensor]]:
        """Fisher-style parameter importances = mean squared ELBO gradient over ``adata``.

        Estimated on an unfrozen copy of ``reference_model`` so every parameter gets a gradient.
        """
        model = deepcopy(reference_model)
        for p in model.module.parameters():
            p.requires_grad = True
        adata = model._validate_anndata(adata)
        scdl = model._make_data_loader(adata=adata, batch_size=256)

        importances = dict(zerolike_params_dict(model.module))
        model.module.eval()
        n_batches = 0
        for tensors in scdl:
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model.module.zero_grad()
            inf = model.module.inference(**model.module._get_inference_input(tensors))
            gen = model.module.generative(**model.module._get_generative_input(tensors, inf))
            loss = model.module.loss(tensors, inf, gen).loss
            loss.backward()
            for name, p in model.module.named_parameters():
                if p.grad is not None and name in importances:
                    importances[name] += p.grad.detach().pow(2)
            n_batches += 1
        n_batches = max(n_batches, 1)
        return [(k, (v / n_batches).detach()) for k, v in importances.items()]

    @classmethod
    def load_query_data_with_replay(
        cls,
        adata: AnnData,
        reference_model: CytoANVI,
        replay_adata: AnnData,
        control_adata: AnnData | None = None,
        combine_type: str = "product",
        freeze_classifier: bool = True,
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
        load_query_kwargs
            Additional keyword args for :meth:`~scvi.model.base.ArchesMixin.load_query_data`.

        Notes
        -----
        - ``ewc_importance`` (= lambda) is set at train time and is dataset-dependent (it scales
          against the Fisher magnitudes); tune it. ``0`` disables the EWC penalty (replay only).
          The paper used ``replay = 0.2`` (buffer fraction) and ``EWC = 100`` for scANVI/RNA;
          CytoVI's intensity likelihood has different Fisher magnitudes, so ``lambda`` must be
          retuned here rather than copied.
        - The continual state (``importances``, ``old_params``, replay batches) is stored as module
          attributes and is **not** in the ``state_dict``: a continual model saved and reloaded
          loses it. Perform the continual update within one session.
        - Control importances are computed on the batch-extended query model (controls may carry
          query batches); reference-replay importances on the reference model.
        """
        if combine_type not in ("additive", "product"):
            raise ValueError("combine_type must be 'additive' or 'product'.")
        if control_adata is None:
            raise ValueError(
                "control_adata is required: the paper's EWC term is F_reference o F_query_ctrl, so "
                "query control cells must be provided. Controls should exist in both reference and "
                "query."
            )

        model = cls.load_query_data(
            adata, reference_model, freeze_classifier=freeze_classifier, **load_query_kwargs
        )

        # Snapshot reference parameter values to anchor to. Taken from the *reference* module (not
        # the surgical query module) so the saved tensors share the reference's shapes and stay
        # aligned with the importances; params resized by surgery (e.g. new batch dims) are skipped
        # in the penalty via a size guard. The penalty compares these against the live query params.
        model.module.old_params = [
            (k, p.detach().clone()) for k, p in reference_model.module.named_parameters()
        ]
        # EWC weight F = F_reference o F_query_ctrl (Hadamard).
        # F_reference: Fisher over the replay buffer (reference cells) -> on the reference model.
        model.module.importances = cls._compute_importances(reference_model, replay_adata)
        # F_query_ctrl: Fisher over query control cells. Controls may carry query batches, so this
        # is computed on the (batch-extended) query model, not the reference.
        model.module.ctrl_importances = cls._compute_importances(model, control_adata)
        model.module.combine_type = combine_type

        # Experience Replay: store replay-buffer minibatches (reference cells) so the training plan
        # can rehearse their ELBO alongside the query loss (paper's L(theta)_{x_query, x_replay}).
        replay_val = model._validate_anndata(replay_adata)
        replay_dl = model._make_data_loader(adata=replay_val, batch_size=256, shuffle=True)
        model.module._replay_batches = [
            {k: v.detach().cpu() for k, v in tensors.items()} for tensors in replay_dl
        ]

        # route training through the EWC + experience-replay plan
        model._training_plan_cls = CytoANVIContinualTrainingPlan
        return model

    @torch.inference_mode()
    def get_uncertainty(
        self,
        adata: AnnData | None = None,
        indices=None,
        batch_size: int | None = None,
        tta_rep: int = 50,
    ) -> np.ndarray:
        """Per-cell Bregman-Information uncertainty via test-time augmentation.

        High scores flag cells whose latent embedding is unstable under feature masking — a proxy
        for novelty / out-of-distribution query cells (e.g. disease-specific states absent from the
        reference). Useful before trusting :meth:`predict` on a mapped query.

        ``tta_rep`` is the number of TTA augmentations used to estimate the Bregman Information;
        more reps give a more stable estimate at linear cost. The default (50) balances stability
        and cost; the paper used ~200.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        scores = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            scores.append(
                compute_uncertainty_scores(inference_inputs, self.module, tta_rep=tta_rep)
            )
        return torch.cat(scores).numpy()
