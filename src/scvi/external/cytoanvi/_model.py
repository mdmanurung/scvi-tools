from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

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
        Prior over labels; if ``None``, uniform.
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
        y_prior=None,
        **model_kwargs,
    ):
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
            y_prior=y_prior,
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
