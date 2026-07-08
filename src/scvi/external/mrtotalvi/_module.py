"""MrTotalVAE — TotalVI with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import LossOutput, auto_move_data

from ._components import EncoderUZ


class MrTotalVAE(TOTALVAE):
    """TotalVI VAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's per-sample attention residual onto TotalVI.  The
    stock TotalVI encoder output becomes the sample-*unaware* base ``u``;
    a donor-specific residual ``eps`` is computed by attending over a
    per-sample embedding table via :class:`~.EncoderUZ`, giving:

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_{\\text{sample}})

    Because ``n_latent_u`` is left at its default (isomorphic), ``z_base = u``
    and the decoder input dimension is identical to stock TotalVI — no decoder
    changes are needed.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| N(0,1))`` — TotalVI's existing term (unchanged).
    * ``kl_z = -\\log p(\\varepsilon) = -\\log N(0, \\exp(\\text{pz\\_scale}))`` — new term.

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
        """
        if n_latent_sample is None:
            n_latent_sample = self._n_latent_sample
        if z_u_prior_scale is None:
            z_u_prior_scale = self._z_u_prior_scale
        if learn_z_u_prior_scale is None:
            learn_z_u_prior_scale = self._learn_z_u_prior_scale

        self._n_sample = n_sample

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder unchanged
        self.qz = EncoderUZ(
            n_latent=self.n_latent,
            n_sample=n_sample,
            n_latent_u=None,
            n_latent_sample=n_latent_sample,
            use_map=True,
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(self.n_latent))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

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

        Calls TotalVI's ``_regular_inference`` to obtain the sample-unaware
        base ``u`` (TotalVI's ``z``), then applies :class:`~.EncoderUZ` to
        decompose it into a sample-aware ``z``:

        .. code-block::

            u = EncoderTOTALVI(x, y)          # TotalVI encoder, unchanged
            z_base, eps = qz(u, sample)        # attention over per-sample embed
            z = z_base + eps                   # → decoder input

        The returned dict replaces ``Z_KEY`` with ``z`` and adds ``"u"``,
        ``"z_base"``, and ``"eps"`` for the loss and counterfactual utilities.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Injected automatically by :meth:`_get_inference_input`.
        cf_sample
            Counterfactual donor override.  If not ``None``, replaces
            ``sample_index`` in the EncoderUZ forward pass.  Used by
            :meth:`~MrTotalVI.compute_local_statistics`.
        """
        # TotalVI encoder → u (sample-unaware Gaussian sample)
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

        # u: shape (batch, n_latent) or (n_samples, batch, n_latent)
        u = out[MODULE_KEYS.Z_KEY]

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        z_base, eps = self.qz(u, sample_index_cf)
        z = z_base + eps

        out[MODULE_KEYS.Z_KEY] = z
        out["u"] = u
        out["z_base"] = z_base
        out["eps"] = eps
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
        peps = Normal(torch.zeros_like(eps), torch.exp(self.pz_scale))
        # kl_z shape: (batch, n_latent) → (batch,)  [or (n_samples, batch) → (batch,)]
        kl_z = -peps.log_prob(eps).sum(dim=-1)
        if kl_z.ndim > 1:
            # n_samples > 1: average over the mc dimension
            kl_z = kl_z.mean(dim=0)

        # Update kl_local: kl_div_z now = kl_u + kl_z
        kl_local = dict(loss_out.kl_local)
        kl_u = kl_local["kl_div_z"]
        kl_local["kl_div_z"] = kl_u + kl_z

        return LossOutput(
            loss=loss_out.loss + kl_weight * kl_z.mean(),
            reconstruction_loss=loss_out.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=loss_out.extra_metrics,
        )
