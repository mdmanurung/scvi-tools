"""MrTotalVI — TotalVI with per-donor hierarchical latent space."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.model._totalvi import TOTALVI
from scvi.model.base import EmbeddingMixin
from scvi.module._constants import MODULE_KEYS
from scvi.utils import setup_anndata_dsp

from ._contracts import (
    ordered_indices_sha256,
    resolve_u_prior,
    resolve_u_prior_supervision,
    take_matrix_rows,
    validate_anndata_counts,
    validate_sample_metadata,
)
from ._counterfactual import (
    get_counterfactual_expression as _get_counterfactual_expression,
)
from ._counterfactual import get_counterfactual_latent as _get_counterfactual_latent
from ._counterfactual import local_sample_enrichment as _local_sample_enrichment
from ._module import MrTotalVAE
from ._stats import (
    _differential_expression,
)
from ._stats import (
    differential_abundance as _differential_abundance,
)
from ._stats import (
    get_aggregated_posterior as _get_aggregated_posterior,
)
from ._stats import (
    get_outlier_cell_sample_pairs as _get_outlier_cell_sample_pairs,
)

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from anndata import AnnData

class MrTotalVI(EmbeddingMixin, TOTALVI):
    """TotalVI with an MrVI-style hierarchical donor latent space.

    Grafts MrVI's per-sample attention residual onto TotalVI, enabling cell-
    and donor-level variability to be **jointly modelled** rather than treating
    donor identity as a nuisance covariate.

    * A mode-selectable u-encoder (:class:`~._components.EncoderXU_TotalVI`)
      produces the base ``u``. The backward-compatible default is
      sample-conditioned; ``u_encoder_mode="sample_blind"`` bypasses biological
      sample conditioning while retaining registered technical covariates.
    * A donor-specific residual ``eps`` is computed via
      :class:`~._components.EncoderUZ` (attention over a per-donor embedding
      table), giving ``z = z_base + eps``.
    * The decoder is **unchanged**; ``z`` drops in with the same shape as
      TotalVI's ``z``.

    ``hierarchy_mode="centered_v2"`` evaluates every registered sample residual
    and subtracts their equal-sample mean before factual gathering or public
    counterfactual decoding. These outputs are registered-sample model
    transformations, not causal interventions.

    Parameters
    ----------
    adata
        AnnData registered via :meth:`~MrTotalVI.setup_anndata`.
    sample_key
        Key in ``adata.obs`` identifying the donor/sample for each cell.
    n_latent
        Dimensionality of the latent space (same as TotalVI).
    n_latent_sample
        Dimension of the per-donor embedding in :class:`~._components.EncoderUZ`.
    z_u_prior_scale
        Log-scale of the prior on ``eps`` (the donor residual).
        ``0.0`` → ``p(eps) = N(0, 1)``.
    learn_z_u_prior_scale
        Whether ``pz_scale`` is a learnable parameter.
    u_prior
        Resolved prior enum: exactly ``"standard"``, ``"mog"``, or ``"vamp"``.
        New calls default to ``"mog"``.
    u_prior_mixture
        Deprecated checkpoint-migration input. Only the exact combinations in
        the 0.2 migration table are accepted; new calls should leave it ``None``.
    u_prior_supervision
        Resolved supervision mode. Omitted/``None`` with zero weight resolves
        to ``"none"``. ``"labels"`` is explicit opt-in and requires registered
        labels plus a finite positive weight.
    u_prior_label_weight
        Label-conditioned mixture-logit weight. Defaults to ``0.0``.
    u_prior_init_seed
        Deterministic seed for training-only Vamp data initialization.
    hierarchy_mode
        ``"legacy"`` preserves historical numerics. ``"centered_v2"`` opts
        into full-registered-universe residual centering and requires
        ``use_map=True`` and ``z_u_prior=True``.
    u_encoder_mode
        ``"sample_conditioned"`` preserves the historical encoder.
        ``"sample_blind"`` bypasses conditional-normalization affine embeddings
        and the explicit sample embedding without changing checkpoint topology.
    kl_u_weight
        Static scalar weight applied to ``KL(q_u ‖ p_u)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    kl_z_weight
        Static scalar weight applied to ``KL(q_z ‖ p_z)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    init_prior_from_data
        If ``True`` and ``u_prior="vamp"``, run k-means on a random subsample
        (≤10 000 cells) of the raw encoder input from the frozen training split
        only. The seed and ordered training-index digest are persisted before
        optimization. Non-Vamp use raises.
    freeze_prior_after_init
        If ``True`` and ``u_prior="vamp"``, freeze the VampPrior pseudo-input
        parameters after data-driven initialisation so they do not drift during
        training.  This configuration is an unvalidated candidate; no
        differential-abundance stability improvement is established.
    use_batch_norm
        Where to apply batch normalisation (``"encoder"``, ``"decoder"``,
        ``"both"``, ``"none"``).  Defaults to ``"none"``; layer normalisation
        is preferred for sample-level models to avoid confounding donor effects
        with batch statistics.
    use_layer_norm
        Where to apply layer normalisation.  Defaults to ``"both"``.
    batch_representation
        How the batch covariate is fed to the networks.  ``"one-hot"`` (default) is the
        historical behaviour and is bit-for-bit unchanged.  ``"embedding"`` replaces the
        one-hot encoding with a single learned embedding table shared by the u-encoder and
        the decoder, so their input width grows by the embedding dimension rather than by
        ``n_batch`` — useful when there are many batches.  Per-batch parameter tables
        (dispersion, per-batch efficiency, protein background prior, library-size priors)
        remain one-hot indexed in both modes, matching :class:`~scvi.module.VAE`.
        Retrieve the learned vectors with :meth:`get_batch_representation`.

        ``EXPERIMENTAL``: existing checkpoints are unaffected, but ``"embedding"`` changes
        the architecture, so a model trained with it cannot be loaded as ``"one-hot"``.
    batch_embedding_kwargs
        Keyword arguments passed to :class:`~scvi.nn.Embedding` when
        ``batch_representation="embedding"``, e.g. ``{"embedding_dim": 5}``.
    **model_kwargs
        Additional keyword arguments forwarded to :class:`~._module.MrTotalVAE`
        (and transitively to :class:`~scvi.module._totalvae.TOTALVAE`).

    Notes
    -----
    ``latent_distribution`` is **forced** to ``"normal"``.  Under ``"ln"``,
    the additive u→z hierarchy is mathematically invalid.

    Three separate categorical axes are kept distinct:
    * ``sample_key`` → :class:`~._components.EncoderUZ` embedding table.
    * ``batch_key``  → decoder ``cat_list``.
    * ``panel_key``  → protein background prior.

    See Also
    --------
    :class:`~._module.MrTotalVAE`
    """

    _module_cls = MrTotalVAE

    def __init__(
        self,
        adata: AnnData,
        sample_key: str,
        n_latent: int = 20,
        n_latent_sample: int = 16,
        n_latent_u: int | None = None,
        z_u_prior: bool = True,
        z_u_prior_scale: float = 0.0,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool | None = None,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 0.0,
        u_prior: str = "mog",
        u_prior_supervision: Literal["none", "labels"] | None = None,
        u_prior_init_seed: int = 0,
        learn_z_u_prior_scale: bool = False,
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        hierarchy_mode: Literal["legacy", "centered_v2"] = "legacy",
        u_encoder_mode: Literal["sample_conditioned", "sample_blind"] = "sample_conditioned",
        scale_observations: bool = False,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
        init_prior_from_data: bool = False,
        freeze_prior_after_init: bool = False,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        batch_embedding_kwargs: dict | None = None,
        **model_kwargs,
    ) -> None:
        resolved_u_prior, resolved_u_prior_mixture = resolve_u_prior(
            u_prior,
            u_prior_mixture,
        )
        manager = self._get_most_recent_anndata_manager(adata, required=True)
        has_registered_labels = manager.registry["setup_args"].get("labels_key") is not None
        resolved_supervision, resolved_label_weight = resolve_u_prior_supervision(
            u_prior_supervision,
            u_prior_label_weight,
            has_registered_labels=has_registered_labels,
            legacy_checkpoint_hint=u_prior_mixture is not None,
            resolved_prior=resolved_u_prior,
        )
        resolved_u_prior_mixture_k = int(u_prior_mixture_k)
        registered_n_labels = int(manager.summary_stats.get("n_labels", 0))
        if (
            u_prior_mixture is not None
            and resolved_u_prior == "vamp"
            and registered_n_labels > 1
        ):
            # Historical labelled Vamp checkpoints used one global mixture
            # with K equal to the registered label count. Preserve both K and
            # the unconditioned (one-dimensional) categorical weights.
            resolved_u_prior_mixture_k = registered_n_labels
        if not isinstance(u_prior_init_seed, int) or isinstance(u_prior_init_seed, bool):
            raise TypeError("u_prior_init_seed must be an integer.")
        if init_prior_from_data and resolved_u_prior != "vamp":
            raise ValueError("init_prior_from_data=True requires u_prior='vamp'.")
        if freeze_prior_after_init and (
            resolved_u_prior != "vamp" or not init_prior_from_data
        ):
            raise ValueError(
                "freeze_prior_after_init=True requires data-initialized u_prior='vamp'."
            )
        if hierarchy_mode not in {"legacy", "centered_v2"}:
            raise ValueError("hierarchy_mode must be one of {'legacy', 'centered_v2'}.")
        if u_encoder_mode not in {"sample_conditioned", "sample_blind"}:
            raise ValueError(
                "u_encoder_mode must be one of {'sample_conditioned', 'sample_blind'}."
            )
        if hierarchy_mode == "centered_v2" and not use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if hierarchy_mode == "centered_v2" and not z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")
        if model_kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrTotalVI requires latent_distribution='normal'. "
                "Under 'ln' (softmax normalisation), the additive u→z "
                "hierarchy is mathematically invalid."
            )
        model_kwargs["latent_distribution"] = "normal"

        # Call TOTALVI.__init__. n_sample / n_latent_sample / z_u_prior_scale /
        # learn_z_u_prior_scale are explicit kwargs here but are absent from
        # TOTALVI's signature — Python routes them into **model_kwargs so they
        # reach MrTotalVAE.__init__ as explicit parameters (not via **kwargs
        # pass-through: MrTotalVAE names them explicitly in its own __init__).
        # MrTotalVAE always builds with n_sample=0 until _setup_hierarchy below
        # so that summary_stats.n_sample (populated by super()) is the source of truth.
        super().__init__(
            adata,
            n_latent=n_latent,
            n_latent_u=n_latent_u,
            z_u_prior=z_u_prior,
            u_prior_scale=u_prior_scale,
            u_prior_mixture=resolved_u_prior_mixture,
            u_prior_mixture_k=resolved_u_prior_mixture_k,
            u_prior_label_weight=resolved_label_weight,
            u_prior=resolved_u_prior,
            u_prior_supervision=resolved_supervision,
            qz_kwargs=qz_kwargs,
            qu_kwargs=qu_kwargs,
            hierarchy_mode=hierarchy_mode,
            u_encoder_mode=u_encoder_mode,
            kl_u_weight=kl_u_weight,
            kl_z_weight=kl_z_weight,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            batch_representation=batch_representation,
            batch_embedding_kwargs=batch_embedding_kwargs,
            **model_kwargs,
        )

        # At this point self.summary_stats is populated from the registry.
        n_sample = self.summary_stats.n_sample

        counts = (
            adata.obs["_scvi_sample"]
            .value_counts()
            .reindex(range(n_sample), fill_value=0)
            .sort_index()
        )
        if (counts == 0).any():
            missing = counts[counts == 0].index.tolist()
            raise ValueError(
                f"Donors at index positions {missing} have no cells in the registered "
                "AnnData. Remove them from sample_key or filter them from the data."
            )
        n_obs_per_sample = torch.tensor(counts.values, dtype=torch.float32)

        self.module._setup_hierarchy(
            n_sample=n_sample,
            n_latent_sample=n_latent_sample,
            z_u_prior_scale=z_u_prior_scale,
            learn_z_u_prior_scale=learn_z_u_prior_scale,
            use_map=use_map,
            hierarchy_mode=hierarchy_mode,
            u_encoder_mode=u_encoder_mode,
            scale_observations=scale_observations,
            n_obs_per_sample=n_obs_per_sample,
            n_labels=self.summary_stats.get("n_labels", 0),
            prior_centroids=None,
            freeze_prior_after_init=False,
        )

        # Sample-level metadata for coordinate labelling
        self._sample_key = sample_key
        self.sample_key = sample_key
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self.resolved_u_prior = resolved_u_prior
        self.u_prior_supervision = resolved_supervision
        self.u_prior_label_weight = resolved_label_weight
        self.u_prior_init_seed = u_prior_init_seed
        self._init_prior_from_data = bool(init_prior_from_data)
        self._freeze_prior_after_init = bool(freeze_prior_after_init)
        self.vamp_training_indices_sha256_ = None
        self.vamp_initialization_seed_ = (
            u_prior_init_seed if init_prior_from_data else None
        )
        self.sample_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        )
        self.label_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        )
        self.sample_info = self.adata.obs[[sample_key]].drop_duplicates().reset_index(drop=True)

        # Overwrite init_params_ from TOTALVI with MrTotalVI's full local scope
        self.init_params_ = self._get_init_params(locals())
        self.init_params_["non_kwargs"].update(
            {
                "u_prior": resolved_u_prior,
                "u_prior_mixture": None,
                "u_prior_supervision": resolved_supervision,
                "u_prior_label_weight": resolved_label_weight,
                "u_prior_mixture_k": resolved_u_prior_mixture_k,
                "u_prior_init_seed": u_prior_init_seed,
            }
        )
        self._refresh_model_summary()

    def _refresh_model_summary(self) -> None:
        """Render the resolved scientific semantics, including checkpoint metadata."""
        self._model_summary_string = (
            "MrTotalVI Model\n"
            f"  hierarchy_mode: {self.hierarchy_mode}, "
            f"u_encoder_mode: {self.u_encoder_mode}\n"
            f"  n_latent: {self.module.n_latent}, "
            f"n_latent_u: {self.module.n_latent_u}, "
            f"n_latent_sample: {self.module._n_latent_sample}\n"
            f"  n_sample: {self.summary_stats.n_sample}, "
            f"z_u_prior: {self.module.z_u_prior}\n"
            f"  u_prior: {self.resolved_u_prior}, "
            f"u_prior_mixture_k: {self.module.resolved_u_prior_mixture_k}\n"
            f"  u_prior_supervision: {self.u_prior_supervision}, "
            f"u_prior_label_weight: {self.u_prior_label_weight}\n"
            f"  vamp_initialization_seed: {self.vamp_initialization_seed_}, "
            "vamp_training_indices_sha256: "
            f"{self.vamp_training_indices_sha256_}"
        )

    def _initialize_vamp_from_training_indices(self, train_indices) -> None:
        """Initialize Vamp pseudo-inputs from the frozen training split only."""
        if not self._init_prior_from_data:
            return

        from sklearn.cluster import KMeans

        ordered_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
        if ordered_indices.size == 0:
            raise ValueError("Cannot initialize VampPrior from an empty training split.")
        if len(np.unique(ordered_indices)) != ordered_indices.size:
            raise ValueError("VampPrior training indices must be unique.")
        if (ordered_indices < 0).any() or (ordered_indices >= self.adata.n_obs).any():
            raise ValueError("VampPrior training indices are out of AnnData bounds.")

        digest = ordered_indices_sha256(ordered_indices)
        if self.vamp_training_indices_sha256_ is not None:
            if self.vamp_training_indices_sha256_ != digest:
                raise RuntimeError(
                    "VampPrior was already initialized from a different training "
                    "boundary; continuing would silently change checkpoint semantics."
                )
            if self._freeze_prior_after_init:
                self.module.u_vamp_pseudo.requires_grad_(False)
            return

        rng = np.random.default_rng(self.u_prior_init_seed)
        selected = rng.choice(
            ordered_indices,
            min(ordered_indices.size, 10_000),
            replace=False,
        )
        n_components = self.module.resolved_u_prior_mixture_k
        if selected.size < n_components:
            raise ValueError(
                "The frozen training split has fewer cells than VampPrior components: "
                f"{selected.size} < {n_components}."
            )

        genes = take_matrix_rows(
            self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY),
            selected,
        ).astype(np.float32, copy=False)
        if self.module.n_input_proteins:
            proteins = take_matrix_rows(
                self.adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY),
                selected,
            ).astype(np.float32, copy=False)
            combined = np.hstack([genes, proteins])
        else:
            combined = genes

        kmeans = KMeans(
            n_clusters=n_components,
            random_state=self.u_prior_init_seed,
            n_init="auto",
        )
        kmeans.fit(combined)
        centroids = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)
        safe = centroids.clamp(min=1e-6, max=20.0)
        pseudo = torch.where(
            centroids > 20.0,
            centroids,
            torch.log(torch.expm1(safe)),
        )
        if not torch.isfinite(pseudo).all():
            raise RuntimeError(
                "Training-only VampPrior initialization produced non-finite values."
            )
        with torch.no_grad():
            self.module.u_vamp_pseudo.copy_(
                pseudo.to(
                    device=self.module.u_vamp_pseudo.device,
                    dtype=self.module.u_vamp_pseudo.dtype,
                )
            )
        if self._freeze_prior_after_init:
            self.module.u_vamp_pseudo.requires_grad_(False)

        self.vamp_training_indices_sha256_ = digest
        self.vamp_initialization_seed_ = self.u_prior_init_seed
        self._refresh_model_summary()

    @classmethod
    def load(
        cls,
        dir_path,
        *args,
        hierarchy_mode_override: Literal["legacy", "centered_v2"] | None = None,
        u_encoder_mode_override: Literal[
            "sample_conditioned", "sample_blind"
        ] | None = None,
        allow_semantic_override: bool = False,
        **kwargs,
    ):
        """Load a checkpoint with explicit, auditable mode overrides.

        Missing mode metadata is handled by the constructor defaults. A
        differing override requires ``allow_semantic_override=True`` because it
        changes model meaning without changing tensor topology.
        """
        if hierarchy_mode_override not in {None, "legacy", "centered_v2"}:
            raise ValueError(
                "hierarchy_mode_override must be one of {None, 'legacy', 'centered_v2'}."
            )
        if u_encoder_mode_override not in {
            None,
            "sample_conditioned",
            "sample_blind",
        }:
            raise ValueError(
                "u_encoder_mode_override must be one of "
                "{None, 'sample_conditioned', 'sample_blind'}."
            )

        model = super().load(dir_path, *args, **kwargs)
        loaded_hierarchy_mode = model.hierarchy_mode
        loaded_u_encoder_mode = model.u_encoder_mode
        resolved_hierarchy_mode = (
            loaded_hierarchy_mode
            if hierarchy_mode_override is None
            else hierarchy_mode_override
        )
        resolved_u_encoder_mode = (
            loaded_u_encoder_mode
            if u_encoder_mode_override is None
            else u_encoder_mode_override
        )
        changed = (
            resolved_hierarchy_mode != loaded_hierarchy_mode
            or resolved_u_encoder_mode != loaded_u_encoder_mode
        )
        if changed and not allow_semantic_override:
            raise ValueError(
                "A differing MrTotalVI mode override changes checkpoint semantics. "
                "Pass allow_semantic_override=True to make the change explicit."
            )
        if resolved_hierarchy_mode == "centered_v2" and not model.module._use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if resolved_hierarchy_mode == "centered_v2" and not model.module.z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")
        if changed:
            warnings.warn(
                "Applying an explicit MrTotalVI semantic override while loading: "
                f"hierarchy_mode {loaded_hierarchy_mode!r} -> "
                f"{resolved_hierarchy_mode!r}; u_encoder_mode "
                f"{loaded_u_encoder_mode!r} -> {resolved_u_encoder_mode!r}.",
                UserWarning,
                stacklevel=2,
            )

        model.loaded_hierarchy_mode = loaded_hierarchy_mode
        model.loaded_u_encoder_mode = loaded_u_encoder_mode
        model.resolved_hierarchy_mode = resolved_hierarchy_mode
        model.resolved_u_encoder_mode = resolved_u_encoder_mode
        model.hierarchy_mode = resolved_hierarchy_mode
        model.u_encoder_mode = resolved_u_encoder_mode
        model.module.hierarchy_mode = resolved_hierarchy_mode
        model.module.u_encoder_mode = resolved_u_encoder_mode
        model.module.qu.u_encoder_mode = resolved_u_encoder_mode
        if (
            model._freeze_prior_after_init
            and model.vamp_training_indices_sha256_ is not None
        ):
            model.module.u_vamp_pseudo.requires_grad_(False)
        model.init_params_["non_kwargs"]["hierarchy_mode"] = resolved_hierarchy_mode
        model.init_params_["non_kwargs"]["u_encoder_mode"] = resolved_u_encoder_mode
        model._refresh_model_summary()
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 4e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        adversarial_classifier: bool | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        external_indexing: list[np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        """Train MrTotalVI after freezing any data-derived prior boundary.

        Data-initialized Vamp pseudo-inputs are resolved from the exact training
        indices before the first optimizer step. Pass ``accelerator='cpu'``
        explicitly to suppress the no-GPU warning.
        """
        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "gpu"
            else:
                warnings.warn(
                    "No CUDA device found — MrTotalVI will train on CPU. "
                    "Pass accelerator='cpu' to suppress this warning, or "
                    "run on a GPU node.",
                    UserWarning,
                    stacklevel=2,
                )

        resolved_external_indexing = external_indexing
        if self._init_prior_from_data:
            split_kwargs = dict(datasplitter_kwargs or {})
            splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size or settings.batch_size,
                external_indexing=external_indexing,
                **split_kwargs,
            )
            splitter.setup()
            resolved_external_indexing = [
                np.asarray(splitter.train_idx, dtype=np.int64),
                np.asarray(splitter.val_idx, dtype=np.int64),
                np.asarray(splitter.test_idx, dtype=np.int64),
            ]
            self._initialize_vamp_from_training_indices(splitter.train_idx)

        return super().train(
            max_epochs=max_epochs,
            lr=lr,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            adversarial_classifier=adversarial_classifier,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            external_indexing=resolved_external_indexing,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        sample_key: str,
        labels_key: str | None = None,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        panel_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            Key in ``adata.obsm`` for raw, finite, non-negative, integer-like
            protein counts. Every value is validated before registration.
        sample_key
            Key in ``adata.obs`` identifying the donor/sample for each cell.
            Each unique value becomes one row in the per-sample embedding table.
        labels_key
            Optional label metadata. Registration alone never changes the
            objective; label supervision requires explicit constructor opt-in.
        protein_names_uns_key
            Key in ``adata.uns`` for protein names.
        %(param_batch_key)s
        panel_key
            Key in ``adata.obs`` for the panel used to measure proteins.
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        validate_anndata_counts(
            adata,
            layer=layer,
            protein_expression_obsm_key=protein_expression_obsm_key,
        )
        # Add integer index column required by compute_local_statistics
        adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)

        setup_method_args = cls._get_setup_method_args(**locals())

        # The panel field mirrors TOTALVI: insert at position 0 when panel_key provided
        if panel_key is not None:
            batch_field = fields.CategoricalObsField("panel", panel_key)
        else:
            batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)

        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            fields.CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            fields.NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
            # MrTotalVI-specific: donor/sample axis
            fields.CategoricalObsField(REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            # Cell indices for batched counterfactual inference
            fields.NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if panel_key is not None:
            # Mirrors TOTALVI: panel field also registered at position 0
            anndata_fields.insert(0, fields.CategoricalObsField("panel", panel_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # u-space statistical APIs
    # ------------------------------------------------------------------

    def get_aggregated_posterior(
        self,
        adata: AnnData | None = None,
        sample: str | int | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
    ):
        """Compute the aggregated posterior mixture over ``u``."""
        return _get_aggregated_posterior(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample=sample,
            indices=indices,
            batch_size=batch_size,
        )

    def differential_abundance(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        compute_log_enrichment: bool = False,
        omit_original_sample: bool = True,
        donor_key: str | None = None,
        batch_size: int = 128,
        n_mc_samples: int = 1,
    ) -> xr.Dataset:
        """Compute descriptive MrVI-style abundance scores over ``u``.

        Parameters
        ----------
        adata
            AnnData to compute DA on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names for grouping.
        sample_subset
            Restrict to these sample names only.
        compute_log_enrichment
            If ``True``, also compute log enrichment vs. the complementary group.
        omit_original_sample
            Exclude a cell's own sample when computing the aggregated log-prob.
        donor_key
            Column in ``adata.obs`` identifying the donor.  When set, per-donor
            log_probs are mean-centred before covariate aggregation, blocking the
            donor effect for repeated-measures designs.
        batch_size
            Dataloader batch size.
        n_mc_samples
            MC draws from ``q(u|x)`` for Jensen-gap bias correction.
            See :func:`~scvi.external.mrtotalvi._stats.differential_abundance`
            for full semantics.  Default ``1`` preserves deterministic behavior.
        """
        adata = self._validate_anndata(adata)
        selected_samples, _ = validate_sample_metadata(
            adata.obs,
            sample_key=self.sample_key,
            covariate_keys=list(sample_cov_keys or []),
            donor_key=donor_key,
            sample_subset=sample_subset,
            authoritative_order=self.sample_order,
        )
        warnings.warn(
            "differential_abundance() returns descriptive, non-inferential "
            "model scores. Use a separately validated replicate-aware method "
            "for biological abundance inference.",
            UserWarning,
            stacklevel=2,
        )
        result = _differential_abundance(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample_cov_keys=sample_cov_keys,
            sample_subset=sample_subset,
            compute_log_enrichment=compute_log_enrichment,
            omit_original_sample=omit_original_sample,
            donor_key=donor_key,
            batch_size=batch_size,
            n_mc_samples=n_mc_samples,
            validated_sample_order=selected_samples,
        )
        result.attrs.update(
            {
                "interpretation": "descriptive_non_inferential",
                "biological_inference_supported": False,
                "sample_order_contract": "declared_subset_order",
            }
        )
        return result

    def get_outlier_cell_sample_pairs(
        self,
        adata: AnnData | None = None,
        subsample_size: int | None = 5_000,
        quantile_threshold: float = 0.05,
        admissibility_threshold: float = 0.0,
        batch_size: int = 256,
    ) -> xr.Dataset:
        """Compute admissible cell-sample pairs using aggregated posteriors over ``u``."""
        return _get_outlier_cell_sample_pairs(
            self,
            adata=adata,
            sample_key=self.sample_key,
            subsample_size=subsample_size,
            quantile_threshold=quantile_threshold,
            admissibility_threshold=admissibility_threshold,
            batch_size=batch_size,
        )

    def get_counterfactual_latent(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        reference_indices=None,
        support_quantile=0.05,
        admissibility_threshold=0.0,
        batch_size=256,
        target_chunk_size=None,
        random_state=0,
        zarr_path=None,
        zarr_chunks=None,
    ) -> xr.Dataset:
        """Return centered latent transformations for registered samples.

        The returned dataset contains raw posterior draws for ``u``, ``z_base``,
        ``eps_raw``, ``eps_centered``, and ``z``, plus separate support and
        admissibility indicators. ``posterior_mc`` also adds posterior means and
        quantiles without dropping the ``draw`` dimension.

        Notes
        -----
        This method requires ``hierarchy_mode="centered_v2"``. Centering always
        uses the full registered sample universe, even when ``target_samples``
        requests a subset. Targets cannot extrapolate to unregistered samples,
        and outputs are model transformations rather than causal interventions.
        Public streaming/Zarr export is quarantined. Requests estimated above
        512 MiB must be reduced through explicit subsetting.
        """
        if zarr_path is not None or zarr_chunks is not None:
            raise NotImplementedError(
                "Public streaming/Zarr export is quarantined for MrTotalVI. "
                "Use bounded in-memory requests with explicit cell, target, and "
                "feature subsetting."
            )
        return _get_counterfactual_latent(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            reference_indices=reference_indices,
            support_quantile=support_quantile,
            admissibility_threshold=admissibility_threshold,
            batch_size=batch_size,
            target_chunk_size=target_chunk_size,
            random_state=random_state,
            zarr_path=None,
            zarr_chunks=None,
        )

    def get_counterfactual_expression(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        gene_list=None,
        protein_list=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        batch_policy="observed",
        specified_batch=None,
        panel_policy="observed",
        specified_panel=None,
        library_policy="observed",
        specified_library_size=None,
        marginal_reference_indices=None,
        batch_size=256,
        target_chunk_size=None,
        feature_chunk_size=None,
        random_state=0,
        zarr_path=None,
        zarr_chunks=None,
    ) -> xr.Dataset:
        """Return deterministic RNA and protein expectations by registered sample.

        ``batch_policy``, ``panel_policy``, and ``library_policy`` make the
        decoder context explicit. ``observed`` holds the factual context fixed;
        ``specified`` requires registered labels or positive library sizes; and
        ``sample_balanced_marginal`` weights biological samples equally while
        retaining empirical joint technical contexts within each sample.

        Protein component means use decoder parameters analytically and never
        use the decoder's stochastic background-rate sample. Unavailable
        proteins are marked by ``protein_available=False`` and all associated
        protein estimands are ``NaN``.

        Notes
        -----
        This method requires ``hierarchy_mode="centered_v2"`` and is limited to
        registered target samples. It returns non-causal model transformations.
        Public streaming/Zarr export is quarantined. Requests estimated above
        512 MiB must be reduced through explicit subsetting.
        """
        if zarr_path is not None or zarr_chunks is not None:
            raise NotImplementedError(
                "Public streaming/Zarr export is quarantined for MrTotalVI. "
                "Use bounded in-memory requests with explicit cell, target, and "
                "feature subsetting."
            )
        return _get_counterfactual_expression(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            gene_list=gene_list,
            protein_list=protein_list,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            batch_policy=batch_policy,
            specified_batch=specified_batch,
            panel_policy=panel_policy,
            specified_panel=specified_panel,
            library_policy=library_policy,
            specified_library_size=specified_library_size,
            marginal_reference_indices=marginal_reference_indices,
            batch_size=batch_size,
            target_chunk_size=target_chunk_size,
            feature_chunk_size=feature_chunk_size,
            random_state=random_state,
            zarr_path=None,
            zarr_chunks=None,
        )

    def local_sample_enrichment(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        reference_adata=None,
        reference_indices=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        group_key=None,
        contrast=None,
        donor_key=None,
        max_reference_cells_per_sample=None,
        batch_size=256,
        reference_chunk_size=None,
        random_state=0,
    ) -> xr.Dataset:
        """Return descriptive local densities over registered samples.

        Each target density is an equal-component mixture of reference
        posteriors over ``u``. A query cell is removed only from its factual
        sample mixture. ``group_key`` aggregates sample densities with
        equal-sample ``logmeanexp``; ``contrast`` is numerator minus denominator;
        and ``donor_key`` requires exact numerator/denominator pairing.

        These outputs are descriptive and non-inferential. A factual singleton
        has zero retained references, ``finite_support=False``, and ``NaN``
        density.
        """
        return _local_sample_enrichment(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            reference_adata=reference_adata,
            reference_indices=reference_indices,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            group_key=group_key,
            contrast=contrast,
            donor_key=donor_key,
            max_reference_cells_per_sample=max_reference_cells_per_sample,
            batch_size=batch_size,
            reference_chunk_size=reference_chunk_size,
            random_state=random_state,
        )

    def differential_expression(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 128,
        normalize_design_matrix: bool = True,
        mc_samples: int = 50,
        filter_inadmissible_samples: bool = False,
        store_lfc: bool = False,
        donor_key: str | None = None,
        delta: float | None = 0.3,
        lambd: float = 0.0,
        store_baseline: bool = False,
        eps_lfc: float = 1e-6,
        use_vmap: bool = False,
        **filter_samples_kwargs,
    ) -> xr.Dataset:
        """Refuse public cell-level DE/LFC for every MrTotalVI hierarchy.

        Biological differential expression requires a donor-pseudobulk method
        such as PyDESeq2, edgeR, or dreamlet. Historical latent-space machinery
        is retained only in a private reproducibility method and is not a
        calibrated inferential API.
        """
        if use_vmap:
            raise NotImplementedError(
                "use_vmap=True is not implemented for MrTotalVI statistics."
            )
        raise RuntimeError(
            "MrTotalVI.differential_expression() is disabled: legacy and "
            "centered-v2 cell-level p-values/LFC are not validated for biological "
            "inference. Use donor-pseudobulk PyDESeq2, edgeR, or dreamlet."
        )

    def _legacy_differential_expression_for_reproducibility(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 128,
        normalize_design_matrix: bool = True,
        mc_samples: int = 50,
        filter_inadmissible_samples: bool = False,
        store_lfc: bool = False,
        donor_key: str | None = None,
        delta: float | None = 0.3,
        lambd: float = 0.0,
        store_baseline: bool = False,
        eps_lfc: float = 1e-6,
        use_vmap: bool = False,
        **filter_samples_kwargs,
    ) -> xr.Dataset:
        """Run the historical non-inferential estimator for private reproduction.

        Fits a per-cell weighted least-squares linear model on the sample-
        specific residual ``eps_d = z_d - z_base`` (the donor latent shift),
        following Boyeau et al. (2023).

        Parameters
        ----------
        adata
            AnnData to compute DE on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names in ``adata.obs``.
        sample_subset
            Restrict DE to these sample names only.
        indices
            Cell indices to process (default: all cells).
        batch_size
            Dataloader batch size.
        normalize_design_matrix
            Min-max normalize each covariate column to [0, 1].
        mc_samples
            Monte-Carlo draws from ``q(u)``; ``1`` uses the posterior mean.
        filter_inadmissible_samples
            Weight out outlier cell-sample pairs via aggregated-posterior
            admissibility scores (see :meth:`get_outlier_cell_sample_pairs`).
        store_lfc
            If ``True``, compute gene/protein log2-fold-changes (LFC) and store
            ``lfc``, ``lfc_std``, and optionally ``pde`` / ``baseline_expression``
            in the returned dataset.  Feature axis is split into ``gene`` and
            ``protein`` coordinates by this wrapper.
        donor_key
            Column in ``adata.obs`` identifying the donor.  Adds donor dummy
            columns as nuisance covariates (repeated-measures approximation).
        delta
            Effect-size threshold for PDE (``P(|lfc| >= delta)``).  Used only
            when ``store_lfc=True``.  Pass ``None`` to skip PDE.
        lambd
            L2 regularisation for ``X^T M X`` inversion.
        store_baseline
            When ``store_lfc=True``, also store the batch-marginalized baseline
            expression (null decode) as ``baseline_expression``.
        eps_lfc
            Small offset added before log2 to avoid log(0).  Default ``1e-6``.
        use_vmap
            ``True`` is unsupported and fails before validation or inference.
        **filter_samples_kwargs
            Forwarded to :meth:`get_outlier_cell_sample_pairs`.
        """
        if use_vmap:
            raise NotImplementedError(
                "use_vmap=True is not implemented for MrTotalVI statistics."
            )
        if self.hierarchy_mode == "centered_v2":
            raise RuntimeError(
                "The historical differential-expression implementation is not "
                "available for centered_v2 models."
            )
        adata = self._validate_anndata(adata)
        selected_samples, _ = validate_sample_metadata(
            adata.obs,
            sample_key=self.sample_key,
            covariate_keys=list(sample_cov_keys or []),
            donor_key=donor_key,
            sample_subset=sample_subset,
            authoritative_order=self.sample_order,
        )
        ds = _differential_expression(
            self,
            adata=adata,
            sample_cov_keys=list(sample_cov_keys) if sample_cov_keys else [],
            sample_subset=sample_subset,
            indices=indices,
            batch_size=batch_size,
            normalize_design_matrix=normalize_design_matrix,
            mc_samples=mc_samples,
            filter_inadmissible_samples=filter_inadmissible_samples,
            store_lfc=store_lfc,
            donor_key=donor_key,
            delta=delta,
            lambd=lambd,
            store_baseline=store_baseline,
            eps_lfc=eps_lfc,
            use_vmap=use_vmap,
            validated_sample_order=selected_samples,
            **filter_samples_kwargs,
        )
        if store_lfc and "feature" in ds.coords:
            n_genes = self.summary_stats.n_vars
            ds = ds.assign_coords(
                feature=(
                    ["gene"] * n_genes
                    + ["protein"] * (ds.sizes["feature"] - n_genes)
                )
            )
        ds.attrs.update(
            {
                "interpretation": "historical_private_non_inferential",
                "biological_inference_supported": False,
            }
        )
        return ds

    # ------------------------------------------------------------------
    # Latent representation
    # ------------------------------------------------------------------

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        give_z: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """Compute the latent representation of the data.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        give_mean
            If ``True``, uses the posterior mean of ``u`` as the query to
            :class:`~.EncoderUZ`.  When ``give_z=True`` and ``give_mean=True``,
            returns ``z_mean = z_base(u_mean) + eps(u_mean, d)``.
        batch_size
            Minibatch size.
        give_z
            If ``True`` (default), returns the sample-aware ``z = z_base + eps``
            using the cell's actual donor index. If ``False``, returns ``u``.
            Under the legacy default encoder, ``u`` is sample-conditioned; it
            is sample-blind only when explicitly configured as such.

        Returns
        -------
        If ``give_z=True``, an array of shape ``(n_obs, n_latent)``.
        If ``give_z=False``, an array of shape ``(n_obs, n_latent_u)``.

        Notes
        -----
        This method does **not** delegate to
        :meth:`super().get_latent_representation` for the ``give_z=True`` path.
        The base class returns ``qz.loc`` when ``give_mean=True``, and ``qz`` is
        the distribution over ``u`` — silently discarding the donor residual.
        Both paths are implemented manually to avoid this silent wrong-output bug.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        was_training = self.module.training
        self.module.eval()
        results = []
        try:
            for tensors in scdl:
                inf_inputs = self.module._get_inference_input(tensors)
                with torch.inference_mode():
                    out = self.module.inference(**inf_inputs)

                if not give_z:
                    # Return u (sample-conditioned base, posterior mean or sample)
                    rep = out["qz"].loc if give_mean else out["u"]
                else:
                    # give_z=True: return z = z_base + eps using real sample_index.
                    # When give_mean=False, out[Z_KEY] is already z_base + eps from
                    # the reparameterised sample.  When give_mean=True we re-run qz
                    # with u_mean so the returned z uses the posterior mean of u.
                    if give_mean:
                        sample_index = inf_inputs["sample_index"]
                        u_mean = out["qz"].loc
                        with torch.inference_mode():
                            if self.hierarchy_mode == "centered_v2":
                                _, _, _, z_all = (
                                    self.module._all_sample_residuals(u_mean)
                                )
                                rep = self.module._gather_sample(
                                    z_all,
                                    sample_index,
                                )
                            else:
                                z_base, eps, _ = self.module.qz(
                                    u_mean,
                                    sample_index,
                                )
                                rep = z_base + eps
                    else:
                        rep = out[MODULE_KEYS.Z_KEY]

                results.append(rep.detach().cpu().numpy())
        finally:
            self.module.train(was_training)
        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Counterfactual representations (internal)
    # ------------------------------------------------------------------

    def _compute_cf_representations(
        self,
        adata: AnnData,
        indices: npt.ArrayLike | None,
        batch_size: int,
        use_mean: bool = True,
    ) -> tuple[npt.NDArray, list[str]]:
        """Compute counterfactual z for every cell × donor combination.

        For each cell, runs the encoder once to get ``qz``, then loops over
        all ``n_sample`` donors.  For each donor ``d``:

        .. code-block::

            cf_sample = d * ones(n_cells)
            z_base, eps = module.qz(u, cf_sample)
            z[cell, d, :] = z_base + eps

        Parameters
        ----------
        adata
            Validated AnnData.
        indices
            Observation indices to use.
        batch_size
            Cells per DataLoader minibatch.
        use_mean
            If ``True``, uses ``qz.loc`` (posterior mean of the encoder) as
            ``u``; otherwise uses the sampled ``u`` from inference.

        Returns
        -------
        reps : np.ndarray
            Shape ``(n_cells, n_sample, n_latent)``.
        cell_names : list[str]
            Original cell names in the same order as ``reps``.
        """
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        n_sample = self.summary_stats.n_sample

        all_reps = []
        all_cell_names = []

        was_training = self.module.training
        self.module.eval()
        try:
            with torch.inference_mode():
                for tensors in tqdm(scdl, desc="Counterfactual representations"):
                    cell_idx = tensors[REGISTRY_KEYS.INDICES_KEY].long().flatten()
                    all_cell_names.extend(adata.obs_names[cell_idx.numpy()].tolist())

                    # Full TotalVI + MrTotalVI encoder pass
                    inf_inputs = self.module._get_inference_input(tensors)
                    base_out = self.module.inference(**inf_inputs)

                    # u: sample-conditioned base (conditioned on real donor)
                    if use_mean:
                        u = base_out["qz"].loc  # posterior mean of qu: (batch, n_latent)
                    else:
                        u = base_out["u"]       # reparameterised sample from qu

                    n_cells = u.shape[0]
                    dev = u.device

                    if self.hierarchy_mode == "centered_v2":
                        _, _, _, z_all = self.module._all_sample_residuals(u)
                        batch_reps = z_all.detach().cpu().numpy()
                    else:
                        # Preserve the legacy counterfactual path exactly.
                        cf_zs = []
                        for d in range(n_sample):
                            cf_sample = torch.full(
                                (n_cells, 1),
                                d,
                                dtype=torch.long,
                                device=dev,
                            )
                            z_base, eps, _ = self.module.qz(u, cf_sample)
                            cf_zs.append((z_base + eps).detach().cpu())

                        batch_reps = (
                            torch.stack(cf_zs, dim=0)
                            .permute(1, 0, 2)
                            .numpy()
                        )
                    all_reps.append(batch_reps)
        finally:
            self.module.train(was_training)

        return np.concatenate(all_reps, axis=0), all_cell_names

    # ------------------------------------------------------------------
    # Public counterfactual API
    # ------------------------------------------------------------------

    def get_local_sample_representation(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
        use_mean: bool = True,
    ) -> xr.DataArray:
        """Compute the local per-donor latent representation.

        For each cell, returns a ``(n_sample, n_latent)`` matrix of
        counterfactual ``z`` values — one per registered donor.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        batch_size
            Minibatch size.
        use_mean
            If ``True``, uses the posterior mean of the encoder as ``u``.

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_latent)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        reps, cell_names = self._compute_cf_representations(
            adata, indices, batch_size, use_mean=use_mean
        )
        return xr.DataArray(
            reps,
            dims=["cell_name", "sample", "latent_dim"],
            coords={
                "cell_name": cell_names,
                "sample": self.sample_order,
            },
            name="sample_representations",
        )

    def get_local_sample_distances(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
        use_mean: bool = True,
        norm: Literal["l2", "l1"] = "l2",
    ) -> xr.DataArray:
        """Compute cell-specific pairwise donor distance matrices.

        For each cell, computes a symmetric ``(n_sample, n_sample)`` distance
        matrix over the counterfactual donor representations.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        batch_size
            Minibatch size.
        use_mean
            If ``True``, uses the posterior mean of the encoder as ``u``.
        norm
            ``"l2"`` (Euclidean) or ``"l1"`` (Manhattan).

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_sample)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        reps, cell_names = self._compute_cf_representations(
            adata, indices, batch_size, use_mean=use_mean
        )
        reps_t = torch.tensor(reps)  # (n_cells, n_sample, n_latent)

        # Pairwise distances per cell: (n_cells, n_sample, n_sample)
        def _pairwise(rep: torch.Tensor) -> torch.Tensor:
            # rep: (n_sample, n_latent)
            delta = rep.unsqueeze(0) - rep.unsqueeze(1)  # (n_s, n_s, n_latent)
            if norm == "l2":
                return torch.sqrt((delta**2).sum(-1))
            elif norm == "l1":
                return delta.abs().sum(-1)
            else:
                raise ValueError(f"Unsupported norm '{norm}'. Choose 'l2' or 'l1'.")

        dists = torch.vmap(_pairwise)(reps_t).numpy()  # (n_cells, n_sample, n_sample)

        return xr.DataArray(
            dists,
            dims=["cell_name", "sample_x", "sample_y"],
            coords={
                "cell_name": cell_names,
                "sample_x": self.sample_order,
                "sample_y": self.sample_order,
            },
            name="sample_distances",
        )
