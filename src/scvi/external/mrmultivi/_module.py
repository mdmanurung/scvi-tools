"""MrMultiVAE — MULTIVAE with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi.module._multivae import MULTIVAE
from scvi.module.base import LossOutput, auto_move_data

from ..mrtotalvi._components import EncoderUZ, EncoderXU_MultiVI


class MrMultiVAE(MULTIVAE):
    """MULTIVAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto MULTIVAE.  A sample-conditioned
    u-encoder (:class:`~scvi.external.mrtotalvi._components.EncoderXU_MultiVI`)
    maps MULTIVAE's mixed latent through donor-conditioned normalization layers,
    then a donor-specific residual ``eps`` is computed by attending over a
    per-donor embedding table via
    :class:`~scvi.external.mrtotalvi._components.EncoderUZ`:

    .. math::
        u \\sim q_u(u_0,\\, d)
        \\quad\\text{(sample-conditioned, where } u_0 \\text{ is MULTIVAE's mixed latent)}

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_{\\text{sample}})

    The decoder receives ``z`` with the same shape as MULTIVAE's ``z`` —
    no decoder changes are needed.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| N(0,1))`` — from the sample-conditioned u-encoder
      (replaces MULTIVAE's ``kl_divergence_z`` via ``qz_m``/``qz_v`` override).
    * ``kl_z = -\\log p(\\varepsilon)`` — new term added here.

    Parameters
    ----------
    n_input_regions
        Number of input ATAC regions.
    n_input_genes
        Number of input RNA genes.
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
        If ``True``, ``pz_scale`` is a learnable :class:`~torch.nn.Parameter`.
    **kwargs
        Forwarded verbatim to :class:`~scvi.module._multivae.MULTIVAE`.
    """

    def __init__(
        self,
        n_input_regions: int = 0,
        n_input_genes: int = 0,
        n_input_proteins: int = 0,
        n_sample: int = 0,
        n_latent_sample: int = 16,
        z_u_prior_scale: float = 0.0,
        learn_z_u_prior_scale: bool = False,
        use_map: bool = True,
        scale_observations: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            n_input_regions=n_input_regions,
            n_input_genes=n_input_genes,
            n_input_proteins=n_input_proteins,
            **kwargs,
        )

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
        """Build or replace the u→z hierarchy after MULTIVAE is initialised.

        Safe to call after ``super().__init__()`` — e.g. once the registry
        summary stats are available in :meth:`~MrMultiVI.__init__`.

        Parameters
        ----------
        n_sample
            Total number of donors registered in the MuData.
        n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale
            Override stored values; ``None`` → use stored value.
        use_map
            If ``False``, treat eps as stochastic.  ``None`` → use stored value.
        scale_observations
            If ``True``, weight each cell's ELBO by ``1 / n_cells_in_that_sample``.
            ``None`` → use stored value.
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

        # Sample-conditioned u-encoder: mirrors MrVI's EncoderXU, takes MULTIVAE mixed latent
        self.qu = EncoderXU_MultiVI(
            n_input=self.n_latent,
            n_latent=self.n_latent,
            n_sample=n_sample,
        )

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder input unchanged
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
        """Extend MULTIVAE's inference input with the sample index."""
        base = super()._get_inference_input(tensors)
        base["sample_index"] = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        return base

    # ------------------------------------------------------------------
    # Inference: u → (z_base, eps) → z
    # ------------------------------------------------------------------

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None,
        cat_covs: torch.Tensor | None,
        label: torch.Tensor,
        cell_idx: torch.Tensor,
        size_factor: torch.Tensor | None,
        n_samples: int = 1,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls MULTIVAE's ``inference`` to obtain the mixed base ``u0``, then
        runs the sample-conditioned u-encoder and the attention residual block:

        .. code-block::

            u0  = mix_modalities(rna_enc(x), atac_enc(x), ...)  # MULTIVAE, unchanged
            qu  = EncoderXU_MultiVI(u0, real_sample)             # sample-conditioned Normal
            u   = qu.rsample()                                   # (batch, n_latent)
            z_base, eps = qz(u, cf_sample)                       # attention over per-sample embed
            z   = z_base + eps                                   # → decoder input

        ``qz_m`` and ``qz_v`` are replaced with ``qu.loc`` and ``qu.scale**2``
        so that MULTIVAE's ``loss()`` computes ``kl_u = KL(qu, N(0,1))``
        automatically.  Our ``loss()`` override then adds
        ``kl_z = -log p(eps)``.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Always the **real** donor (used by ``qu``).
        cf_sample
            Counterfactual donor override for ``eps`` only.  If not ``None``,
            replaces ``sample_index`` in the :class:`~.EncoderUZ` forward pass.
            ``qu`` always conditions on the real donor.
        """
        outputs = super().inference(
            x, y, batch_index, cont_covs, cat_covs, label, cell_idx, size_factor, n_samples
        )

        if not hasattr(self, "qz"):
            # Hierarchy not yet built (n_sample=0 placeholder).
            return outputs

        # u0: MULTIVAE's mixed base latent — input to the sample-conditioned encoder
        u0 = outputs["z"]  # (batch, n_latent)

        # Sample-conditioned u-encoder: qu(u0, real_sample) → Normal(mu_u, sigma_u)
        qu = self.qu(u0, sample_index)
        u = qu.rsample()  # (batch, n_latent)

        # Replace qz_m / qz_v so MULTIVAE's loss computes KL(qu, N(0,1)) as kl_u
        outputs["qz_m"] = qu.loc
        outputs["qz_v"] = qu.scale ** 2

        # eps residual: cf_sample enables counterfactual donor substitution
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        if sample_index_cf is None:
            raise ValueError(
                "sample_index must be provided when the MrMultiVI hierarchy is active."
            )
        z_base, eps, eps_dist = self.qz(u, sample_index_cf)
        z = z_base + eps

        outputs["u"] = u
        outputs["z_base"] = z_base
        outputs["eps"] = eps
        outputs["eps_dist"] = eps_dist  # None when use_map=True
        outputs["z"] = z
        return outputs

    # ------------------------------------------------------------------
    # Loss: two-level KL (kl_u + kl_z)
    # ------------------------------------------------------------------

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Extend MULTIVAE's loss with the second-level KL for the sample residual.

        MULTIVAE's ``kl_divergence_z = KL(q_u \\| N(0,1))`` is already
        ``kl_u`` verbatim (the encoder distribution is unchanged).  This
        method adds:

        .. math::
            kl_z = -\\log p(\\varepsilon) = -\\log N(0,\\; \\exp(\\text{pz\\_scale}))(\\varepsilon)

        and folds it into both ``loss`` and ``kl_local["kl_divergence_z"]``.

        Note
        ----
        The MULTIVAE key is ``"kl_divergence_z"`` (not ``"kl_div_z"`` which is
        the TOTALVAE key used in :class:`~.MrTotalVAE`).
        """
        loss_out = super().loss(tensors, inference_outputs, generative_outputs, kl_weight)

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
        assert kl_z.ndim == 1, f"Expected 1D kl_z; got shape {kl_z.shape}"

        # kl_divergence_z now = kl_u + kl_z
        kl_local = dict(loss_out.kl_local)
        kl_u = kl_local["kl_divergence_z"]
        kl_local["kl_divergence_z"] = kl_u + kl_z

        if self._scale_observations and self.n_obs_per_sample is not None:
            # Weight each cell's ELBO by 1/n_cells_in_that_sample.
            # Reconstruct per-cell ELBO from component tensors (parent already called mean()).
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64).flatten()
            prefactors = self.n_obs_per_sample[sample_index]  # (batch,)
            recon = (
                loss_out.reconstruction_loss["reconstruction_loss_expression"]
                + loss_out.reconstruction_loss["reconstruction_loss_accessibility"]
                + loss_out.reconstruction_loss["reconstruction_loss_protein"]
            )
            per_cell = recon + kl_weight * kl_local["kl_divergence_z"] + kl_local["kl_divergence_paired"]
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
