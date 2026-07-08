"""MrTotalVAE — TotalVI with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import LossOutput, auto_move_data

from ._components import EncoderUZ, EncoderXU_TotalVI


class MrTotalVAE(TOTALVAE):
    """TotalVI VAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto TotalVI as faithfully as the
    multimodal input allows.  A sample-conditioned u-encoder
    (:class:`~.EncoderXU_TotalVI`) replaces TotalVI's stock encoder for the
    base representation ``u``; a donor-specific residual ``eps`` is computed
    by attending over a per-sample embedding table via :class:`~.EncoderUZ`,
    giving:

    .. math::
        u \\sim q_u(x_{\\text{rna}},\\, x_{\\text{prot}},\\, d)
        \\quad\\text{(sample-conditioned)}

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_d)

    Because ``n_latent_u`` is left at its default (isomorphic), ``z_base = u``
    and the decoder input dimension is identical to stock TotalVI — no decoder
    changes are needed.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| N(0,1))`` — from the sample-conditioned u-encoder.
    * ``kl_z = -\\log p(\\varepsilon) = -\\log N(0, \\exp(\\text{pz\\_scale}))`` — eps residual.

    Parameters
    ----------
    n_input_genes
        Number of input genes.
    n_input_proteins
        Number of input proteins.
    n_sample
        Number of donors/samples.  Pass ``0`` as a placeholder; call
        :meth:`_setup_hierarchy` before training.
    n_latent_sample
        Dimension of the per-sample embedding in :class:`~.EncoderUZ`.
    z_u_prior_scale
        Log-scale of the prior on ``eps``.  ``0.0`` → ``p(eps) = N(0,1)``.
    learn_z_u_prior_scale
        If ``True``, ``pz_scale`` is a learnable :class:`~torch.nn.Parameter`;
        otherwise it is a fixed :func:`~torch.Tensor.register_buffer`.
    **kwargs
        All remaining keyword arguments forwarded verbatim to
        :class:`~scvi.module._totalvae.TOTALVAE`.

    Notes
    -----
    ``latent_distribution`` is **forced** to ``"normal"``.  Under ``"ln"``
    (softmax normalisation), ``u`` is simplex-constrained and the additive
    residual hierarchy is mathematically invalid.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_sample: int = 0,
        n_latent_sample: int = 16,
        z_u_prior_scale: float = 0.0,
        learn_z_u_prior_scale: bool = False,
        use_map: bool = True,
        scale_observations: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrTotalVAE requires latent_distribution='normal'. "
                "Under 'ln' (softmax), u is simplex-constrained and the "
                "additive residual hierarchy is invalid."
            )
        kwargs["latent_distribution"] = "normal"

        super().__init__(n_input_genes, n_input_proteins, **kwargs)

        # Store hyperparams so _setup_hierarchy can use them if called deferred
        self._n_sample = n_sample
        self._n_latent_sample = n_latent_sample
        self._z_u_prior_scale = z_u_prior_scale
        self._learn_z_u_prior_scale = learn_z_u_prior_scale
        self._use_map = use_map
        self._scale_observations = scale_observations

        if n_sample > 0:
            self._setup_hierarchy(n_sample, n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale)

    # ------------------------------------------------------------------
    # Hierarchy setup
    # ------------------------------------------------------------------

    def _setup_hierarchy(
        self,
        n_sample: int,
        n_latent_sample: int | None = None,
        z_u_prior_scale: float | None = None,
        learn_z_u_prior_scale: bool | None = None,
        use_map: bool | None = None,
        scale_observations: bool | None = None,
        n_obs_per_sample: torch.Tensor | None = None,
    ) -> None:
        """Build / replace the u→z hierarchy after the base TOTALVAE is initialised.

        Safe to call from :meth:`~MrTotalVI.__init__` after ``super().__init__``,
        e.g. once the registry summary stats are available.

        Parameters
        ----------
        n_sample
            Total number of donors registered in the AnnData.
        n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale
            Override the values stored in ``__init__``; ``None`` → use stored value.
        use_map
            If ``False``, treat eps as stochastic (split attention output into mean and
            log-scale; reparameterise).  ``None`` → use value from ``__init__``.
        scale_observations
            If ``True``, weight each cell's ELBO contribution by
            ``1 / n_cells_in_that_sample`` so high-cell-count donors do not dominate.
            ``None`` → use value from ``__init__``.
        n_obs_per_sample
            Integer tensor of shape ``(n_sample,)`` with per-donor cell counts.
            Required (and used only) when ``scale_observations=True``.
        """
        if n_latent_sample is None:
            n_latent_sample = self._n_latent_sample
        if z_u_prior_scale is None:
            z_u_prior_scale = self._z_u_prior_scale
        if learn_z_u_prior_scale is None:
            learn_z_u_prior_scale = self._learn_z_u_prior_scale
        if use_map is None:
            use_map = self._use_map
        if scale_observations is None:
            scale_observations = self._scale_observations

        self._n_sample = n_sample
        self._use_map = use_map
        self._scale_observations = scale_observations

        # Sample-conditioned u-encoder: mirrors MrVI's EncoderXU, multimodal input
        self.qu = EncoderXU_TotalVI(
            n_input_genes=self.n_input_genes,
            n_input_proteins=self.n_input_proteins,
            n_latent=self.n_latent,
            n_sample=n_sample,
        )

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder unchanged
        self.qz = EncoderUZ(
            n_latent=self.n_latent,
            n_sample=n_sample,
            n_latent_u=None,
            n_latent_sample=n_latent_sample,
            use_map=use_map,
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(self.n_latent))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

        # Per-sample cell counts for observation reweighting (non-persistent: recomputed at load)
        if scale_observations and n_obs_per_sample is not None:
            self.register_buffer("n_obs_per_sample", n_obs_per_sample.float(), persistent=False)
        else:
            self.n_obs_per_sample = None

    # ------------------------------------------------------------------
    # DataLoader → inference plumbing
    # ------------------------------------------------------------------

    def _get_inference_input(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extend TotalVI's inference input with the sample index.

        The ``SAMPLE_KEY`` tensor is added unconditionally; it is present in
        every DataLoader batch once :meth:`MrTotalVI.setup_anndata` has
        registered it.  Minified-mode inference is not supported for
        MrTotalVAE (the hierarchy requires the full encoder path).
        """
        base = super()._get_inference_input(tensors)
        base["sample_index"] = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        return base

    # ------------------------------------------------------------------
    # Inference: u → (z_base, eps) → z
    # ------------------------------------------------------------------

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        n_samples: int = 1,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls TotalVI's ``_regular_inference`` to obtain library size, protein
        background priors, and dispersion parameters (all of which are parallel
        to ``z`` and require no ``z`` dependency), then runs the sample-conditioned
        u-encoder to replace TotalVI's ``qz`` and ``z``:

        .. code-block::

            qu = EncoderXU_TotalVI(x, y, real_sample)  # sample-conditioned Normal
            u  = qu.rsample()                           # (batch, n_latent) or (mc, batch, n_latent)
            z_base, eps = qz(u, cf_sample)              # attention over per-sample embed
            z  = z_base + eps                           # → decoder input

        ``QZ_KEY`` in the returned dict is replaced with ``qu`` so that
        TotalVI's ``loss()`` computes ``kl_u = KL(qu, N(0,1))`` automatically.
        Our ``loss()`` override then adds ``kl_z = -log p(eps)``.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Used by ``qu``; always the **real** donor (never counterfactual).
        cf_sample
            Counterfactual donor override for ``eps`` only.  If not ``None``,
            replaces ``sample_index`` in the :class:`~.EncoderUZ` forward pass.
            ``qu`` always uses ``sample_index`` (real donor).
        """
        # TotalVI encoder pass — we keep ql, library_gene, back_mean_prior, dispersions.
        # Its qz and z are discarded and replaced below.
        out = super()._regular_inference(
            x, y,
            batch_index=batch_index,
            panel_index=panel_index,
            label=label,
            n_samples=n_samples,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )

        if not hasattr(self, "qz"):
            # Hierarchy not yet built (n_sample=0 placeholder). Return TotalVI outputs.
            return out

        # Sample-conditioned u-encoder: uses real sample_index (not cf_sample)
        qu = self.qu(x, y, sample_index)  # Normal: params (batch, n_latent)
        if n_samples > 1:
            u = qu.rsample((n_samples,))  # (n_samples, batch, n_latent)
        else:
            u = qu.rsample()  # (batch, n_latent)

        # Replace TotalVI's qz with qu so base loss computes KL(qu, N(0,1)) as kl_u
        out[MODULE_KEYS.QZ_KEY] = qu

        # eps residual: cf_sample enables counterfactual donor substitution
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        z_base, eps, eps_dist = self.qz(u, sample_index_cf)
        z = z_base + eps

        out[MODULE_KEYS.Z_KEY] = z
        out["u"] = u
        out["z_base"] = z_base
        out["eps"] = eps
        out["eps_dist"] = eps_dist  # None when use_map=True
        return out

    # ------------------------------------------------------------------
    # Loss: two-level KL (kl_u + kl_z)
    # ------------------------------------------------------------------

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
        generative_outputs: dict[str, torch.Tensor | dict],
        pro_recons_weight: float = 1.0,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Extend TotalVI's loss with the second-level KL for the sample residual.

        TotalVI's existing ``kl_div_z = KL(q_u \\| N(0,1))`` is already
        ``kl_u`` verbatim (the encoder distribution is unchanged).  This
        method adds:

        .. math::
            kl_z = -\\log p(\\varepsilon) = -\\log N(0,\\; \\exp(\\text{pz\\_scale}))(\\varepsilon)

        and folds it into both ``loss`` and ``kl_local["kl_div_z"]``.
        """
        loss_out = super().loss(
            tensors, inference_outputs, generative_outputs,
            pro_recons_weight, kl_weight,
        )

        if "eps" not in inference_outputs:
            # Hierarchy not built (n_sample=0 placeholder).
            return loss_out

        eps = inference_outputs["eps"]
        eps_dist = inference_outputs.get("eps_dist")
        peps = Normal(0.0, torch.exp(self.pz_scale))
        if eps_dist is not None:
            # use_map=False: analytic KL(q(eps) || p(eps)) — correct ELBO includes entropy
            kl_z = kl_divergence(eps_dist, peps).sum(dim=-1)
        else:
            # use_map=True: deterministic eps — cross-entropy -log p(eps) is the correct term
            kl_z = -peps.log_prob(eps).sum(dim=-1)
        # kl_z shape: (batch,) or (n_samples, batch) → reduce mc dim
        if kl_z.ndim > 1:
            kl_z = kl_z.mean(dim=0)

        # Update kl_local: kl_div_z now = kl_u + kl_z
        kl_local = dict(loss_out.kl_local)
        kl_u = kl_local["kl_div_z"]
        kl_local["kl_div_z"] = kl_u + kl_z

        if self._scale_observations and self.n_obs_per_sample is not None:
            # Weight each cell's ELBO by 1/n_cells_in_that_sample so high-cell-count
            # donors do not dominate training.  Reconstruct the full per-cell ELBO from
            # individual component tensors (parent loss() already called torch.mean()).
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64).flatten()
            prefactors = self.n_obs_per_sample[sample_index]  # (batch,)
            per_cell = (
                loss_out.reconstruction_loss["reconst_loss_gene"]
                + kl_weight * pro_recons_weight * loss_out.reconstruction_loss["reconst_loss_protein"]
                + kl_weight * kl_local["kl_div_z"]
                + kl_local["kl_div_l_gene"]
                + kl_weight * kl_local["kl_div_back_pro"]
            )
            loss = (per_cell / prefactors).mean()
        else:
            if self._scale_observations:
                import warnings
                warnings.warn(
                    "scale_observations=True but n_obs_per_sample is None "
                    "(buffer was not restored after load). "
                    "Call _setup_hierarchy(n_obs_per_sample=...) to restore weighting.",
                    UserWarning,
                    stacklevel=2,
                )
            loss = loss_out.loss + kl_weight * kl_z.mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=loss_out.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=loss_out.extra_metrics,
        )
