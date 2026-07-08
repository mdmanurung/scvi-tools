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

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from anndata import AnnData

class MrTotalVI(TOTALVI):
    """TotalVI with an MrVI-style hierarchical donor latent space.

    Grafts MrVI's per-sample attention residual onto TotalVI, enabling cell-
    and donor-level variability to be **jointly modelled** rather than treating
    donor identity as a nuisance covariate.

    * TotalVI's ``EncoderTOTALVI`` output becomes the *sample-unaware* base
      ``u``.
    * A donor-specific residual ``eps`` is computed via
      :class:`~._components.EncoderUZ` (attention over a per-donor embedding
      table), giving ``z = z_base + eps``.
    * The decoder is **unchanged**; ``z`` drops in with the same shape as
      TotalVI's ``z``.

    Counterfactual queries ask "what would cell ``i`` look like in donor
    ``d``?" by substituting donor ``d``'s embedding into the attention block.

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
        z_u_prior_scale: float = 0.0,
        learn_z_u_prior_scale: bool = False,
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
        super().__init__(adata, n_latent=n_latent, **model_kwargs)

        # At this point self.summary_stats is populated from the registry.
        n_sample = self.summary_stats.n_sample
        self.module._setup_hierarchy(
            n_sample=n_sample,
            n_latent_sample=n_latent_sample,
            z_u_prior_scale=z_u_prior_scale,
            learn_z_u_prior_scale=learn_z_u_prior_scale,
        )

        # Sample-level metadata for coordinate labelling
        self._sample_key = sample_key
        self.sample_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        )

        self._model_summary_string = (
            f"MrTotalVI Model\n"
            f"  n_latent: {n_latent}, n_latent_sample: {n_latent_sample}\n"
            f"  n_sample: {n_sample}, z_u_prior_scale: {z_u_prior_scale}\n"
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
            fields.CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
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
                    # Return u (sample-unaware base)
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
                            z_base, eps = self.module.qz(u_mean, sample_index)
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

                    # u: sample-unaware base representation
                    if use_mean:
                        u = base_out["qz"].loc  # posterior mean: (batch, n_latent)
                    else:
                        u = base_out["u"]       # reparameterised sample

                    n_cells = u.shape[0]
                    dev = u.device

                    # Counterfactual loop: fix u, vary donor
                    cf_zs = []
                    for d in range(n_sample):
                        cf_sample = torch.full((n_cells, 1), d, dtype=torch.long, device=dev)
                        z_base, eps = self.module.qz(u, cf_sample)
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
