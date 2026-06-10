from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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
from scvi.model.base._archesmixin import ArchesMixin, _get_loaded_data
from scvi.utils import setup_anndata_dsp

from ._continual import ContinualUpdate, CytoANVIContinualTrainingPlan
from ._module import CytoANVAE
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
    - If the query was measured with a *different antibody panel* than the reference, call
      :meth:`prepare_query_anndata` first: it pads missing markers and masks them via CytoVI's
      ``nan_layer`` (rather than treating padded zeros as observed intensities). This requires the
      reference to have been set up with a ``nan_layer`` (i.e. a genuine backbone /
      panel-specific split). The query must fully observe the reference backbone; only
      panel-specific (non-backbone) markers may be absent.
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
            A trained, **in-memory** :class:`CytoANVI` reference (required for the actual prep, so
            the shared backbone can be verified). A saved path is accepted only with
            ``return_reference_var_names=True``; otherwise load it first with :meth:`load`.
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

        # The actual prep needs the in-memory reference to verify the shared backbone: the encoder
        # marker set (encoder_marker_mask) is not recoverable from saved files alone.
        if isinstance(reference_model, str):
            raise ValueError(
                "Panel-aware prepare_query_anndata needs an in-memory reference model to verify "
                "the shared backbone (the encoder marker set is not recoverable from saved files "
                "alone). Load it first: reference_model = CytoANVI.load(path)."
            )

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

        # Reference backbone = the encoder markers. CytoVI encodes *only* the backbone, and
        # scArches re-derives the query backbone from its nan mask, so the query backbone must
        # match the reference backbone exactly. With a nan_layer set, CytoVI always builds this.
        enc = getattr(reference_model.module, "encoder_marker_mask", None)
        if enc is None or len(enc) != len(ref_var_names):
            raise ValueError(
                "The reference has no usable encoder backbone mask (encoder_marker_mask); "
                "panel-aware prep needs a reference whose nan_layer yields a genuine backbone / "
                "panel-specific split."
            )
        backbone_mask = np.asarray(enc, dtype=bool)

        missing_backbone = ref_var_names[backbone_mask].intersection(missing_markers)
        if len(missing_backbone):
            raise ValueError(
                f"Query panel is missing backbone (encoder) markers {list(missing_backbone)}. "
                "CytoVI encodes only the shared backbone, so the backbone must be present in "
                "both reference and query; only panel-specific (non-backbone) markers may be "
                "absent from the query. Add the missing backbone markers to the query, or use "
                "a reference whose backbone is shared with this query."
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
        - The continual update (:class:`~scvi.external.cytoanvi.ContinualUpdate`) is held by the
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
            reference_model, model, replay_adata, control_adata, combine_type=combine_type
        )
        # Persisted across save/load (replay buffer excluded); reattached in CytoANVAE.on_load.
        model.continual_update_state_ = model.module.continual.persistable_state()

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
