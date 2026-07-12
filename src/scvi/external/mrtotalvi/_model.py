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
from scvi.module._constants import MODULE_KEYS
from scvi.utils import setup_anndata_dsp

from ._module import MrTotalVAE
from ._stats import (
    _differential_expression,
    differential_abundance as _differential_abundance,
    get_aggregated_posterior as _get_aggregated_posterior,
    get_outlier_cell_sample_pairs as _get_outlier_cell_sample_pairs,
)

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from anndata import AnnData

class MrTotalVI(TOTALVI):
    """TotalVI with an MrVI-style hierarchical donor latent space.

    Grafts MrVI's per-sample attention residual onto TotalVI, enabling cell-
    and donor-level variability to be **jointly modelled** rather than treating
    donor identity as a nuisance covariate.

    * A sample-conditioned u-encoder (:class:`~._components.EncoderXU_TotalVI`)
      produces the base ``u ~ q_u(x_rna, x_protein, donor)`` — mirroring MrVI's
      ``EncoderXU`` with multimodal input.
    * A donor-specific residual ``eps`` is computed via
      :class:`~._components.EncoderUZ` (attention over a per-donor embedding
      table), giving ``z = z_base + eps``.
    * The decoder is **unchanged**; ``z`` drops in with the same shape as
      TotalVI's ``z``.

    Counterfactual queries ask "what would cell ``i`` look like in donor
    ``d``?" by substituting donor ``d``'s embedding into the attention block
    while holding ``u`` (from the real donor's encoder) fixed.

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
    kl_u_weight
        Static scalar weight applied to ``KL(q_u ‖ p_u)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    kl_z_weight
        Static scalar weight applied to ``KL(q_z ‖ p_z)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    init_prior_from_data
        If ``True`` and ``u_prior="vamp"``, run k-means on a random subsample
        (≤10 000 cells) of the raw encoder input (genes + proteins) and use the
        centroids — mapped through the softplus-inverse — to initialise VampPrior
        pseudo-inputs near the data manifold (see :cite:t:`Tomczak2018`).
        Ignored for ``u_prior="mog"`` (latent-space centroids require a forward
        pass, not available at init time).
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
        u_prior_mixture: bool = True,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 10.0,
        u_prior: str = "mog",
        learn_z_u_prior_scale: bool = False,
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        scale_observations: bool = False,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
        init_prior_from_data: bool = False,
        freeze_prior_after_init: bool = False,
        **model_kwargs,
    ) -> None:
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
            u_prior_mixture=u_prior_mixture,
            u_prior_mixture_k=u_prior_mixture_k,
            u_prior_label_weight=u_prior_label_weight,
            u_prior=u_prior,
            qz_kwargs=qz_kwargs,
            qu_kwargs=qu_kwargs,
            kl_u_weight=kl_u_weight,
            kl_z_weight=kl_z_weight,
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

        prior_centroids = None
        if init_prior_from_data and u_prior == "vamp":
            from scipy.sparse import issparse
            from sklearn.cluster import KMeans

            rng = np.random.default_rng(0)
            n_cells = adata.n_obs
            idx = rng.choice(n_cells, min(n_cells, 10_000), replace=False)

            X_genes = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[idx]
            if issparse(X_genes):
                X_genes = X_genes.toarray()
            X_genes = X_genes.astype(np.float32)

            n_proteins = self.module.n_input_proteins
            if n_proteins > 0:
                X_prot = self.adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)[idx]
                if issparse(X_prot):
                    X_prot = X_prot.toarray()
                X_combined = np.hstack([X_genes, X_prot.astype(np.float32)])
            else:
                X_combined = X_genes

            kmeans = KMeans(n_clusters=u_prior_mixture_k, random_state=0, n_init="auto")
            kmeans.fit(X_combined)
            c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            # Softplus-inverse: F.softplus(prior_centroids) ≈ data centroids
            prior_centroids = torch.log(torch.expm1(c.clamp(min=1e-6)))

        self.module._setup_hierarchy(
            n_sample=n_sample,
            n_latent_sample=n_latent_sample,
            z_u_prior_scale=z_u_prior_scale,
            learn_z_u_prior_scale=learn_z_u_prior_scale,
            use_map=use_map,
            scale_observations=scale_observations,
            n_obs_per_sample=n_obs_per_sample,
            n_labels=self.summary_stats.get("n_labels", 0),
            prior_centroids=prior_centroids,
            freeze_prior_after_init=freeze_prior_after_init,
        )

        # Sample-level metadata for coordinate labelling
        self._sample_key = sample_key
        self.sample_key = sample_key
        self.sample_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        )
        self.label_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        )
        self.sample_info = self.adata.obs[[sample_key]].drop_duplicates().reset_index(drop=True)

        self._model_summary_string = (
            f"MrTotalVI Model\n"
            f"  n_latent: {n_latent}, n_latent_u: {self.module.n_latent_u}, "
            f"n_latent_sample: {n_latent_sample}\n"
            f"  n_sample: {n_sample}, z_u_prior: {z_u_prior}, "
            f"z_u_prior_scale: {z_u_prior_scale}\n"
            f"  u_prior_mixture: {u_prior_mixture}, "
            f"u_prior_mixture_k: {self.module.resolved_u_prior_mixture_k}\n"
            f"  gene_likelihood: {model_kwargs.get('gene_likelihood', 'nb')}"
        )
        # Overwrite init_params_ from TOTALVI with MrTotalVI's full local scope
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, *args, accelerator: str = "auto", **kwargs) -> None:
        """Train MrTotalVI.

        Identical to :meth:`~scvi.model.TOTALVI.train` but warns when no CUDA
        device is detected instead of silently falling back to CPU.

        Pass ``accelerator='cpu'`` explicitly to suppress the warning and run on CPU.
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
        super().train(*args, accelerator=accelerator, **kwargs)

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
            Key in ``adata.obsm`` for protein expression data.
        sample_key
            Key in ``adata.obs`` identifying the donor/sample for each cell.
            Each unique value becomes one row in the per-sample embedding table.
        labels_key
            Optional key in ``adata.obs`` identifying labels used to condition
            the mixture-of-Gaussians prior over ``u``.
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
    ) -> xr.Dataset:
        """Compute MrVI-style differential abundance log probabilities over ``u``.

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
        """
        return _differential_abundance(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample_cov_keys=sample_cov_keys,
            sample_subset=sample_subset,
            compute_log_enrichment=compute_log_enrichment,
            omit_original_sample=omit_original_sample,
            donor_key=donor_key,
            batch_size=batch_size,
        )

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
        """MrVI-style latent-space differential expression.

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
            Reserved; currently ignored (loop path is always used).
        **filter_samples_kwargs
            Forwarded to :meth:`get_outlier_cell_sample_pairs`.
        """
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
            using the cell's actual donor index.  If ``False``, returns the
            sample-unaware base ``u``.

        Returns
        -------
        Array of shape ``(n_obs, n_latent)``.

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
                            z_base, eps, _ = self.module.qz(u_mean, sample_index)
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

                    # Counterfactual loop: fix u, vary donor
                    cf_zs = []
                    for d in range(n_sample):
                        cf_sample = torch.full((n_cells, 1), d, dtype=torch.long, device=dev)
                        z_base, eps, _ = self.module.qz(u, cf_sample)
                        cf_zs.append((z_base + eps).detach().cpu())

                    # (n_sample, n_cells, n_latent) → (n_cells, n_sample, n_latent)
                    batch_reps = torch.stack(cf_zs, dim=0).permute(1, 0, 2).numpy()
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
