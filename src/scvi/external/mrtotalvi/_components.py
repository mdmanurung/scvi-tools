"""Neural network components for MrTotalVI.

Ported from :mod:`scvi.external.mrvi_torch._components` (AttentionBlock, MLP, etc.)
and :mod:`scvi.external.mrvi_torch._module` (EncoderUZ). These are the shared building
blocks for the u→z hierarchical latent; ported here so MultiVI can reuse EncoderUZ
without depending on mrvi_torch internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from scvi.module.base import auto_move_data


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation, matching JAX/Flax ``nn.gelu(approximate=True)``."""
    return F.gelu(x, approximate="tanh")


class Dense(nn.Linear):
    """Thin alias for :class:`~torch.nn.Linear`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ResnetBlock(nn.Module):
    """Resnet block.

    Parameters
    ----------
    n_in
        Number of input units.
    n_out
        Number of output units.
    n_hidden
        Number of hidden units.
    internal_activation
        Activation function to use after the first :class:`~Dense` layer.
    output_activation
        Activation function to use after the last :class:`~Dense` layer.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        internal_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        output_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.internal_activation = internal_activation
        self.output_activation = output_activation

        self.fc1 = nn.Linear(in_features=n_in, out_features=n_hidden)
        self.layer_norm1 = nn.LayerNorm(n_hidden, eps=1e-6)

        if n_in != n_hidden:
            self.fc_match = nn.Linear(in_features=n_in, out_features=n_hidden)
        else:
            self.fc_match = None

        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_out)
        self.layer_norm2 = nn.LayerNorm(n_out, eps=1e-6)

    @auto_move_data
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.fc1(inputs)
        h = self.layer_norm1(h)
        h = self.internal_activation(h)

        if self.fc_match is not None:
            h = h + self.fc_match(inputs)
        else:
            h = h + inputs

        h = self.fc2(h)
        h = self.layer_norm2(h)
        return self.output_activation(h)


class MLP(nn.Module):
    """Multi-layer perceptron with resnet blocks.

    Applies ``n_layers`` :class:`~ResnetBlock` blocks to the input, followed by a
    :class:`~Dense` layer to project to the output dimension.

    Parameters
    ----------
    n_in
        Number of input units.
    n_out
        Number of output units.
    n_hidden
        Number of hidden units.
    n_layers
        Number of resnet blocks.
    activation
        Activation function to use.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation

        self.resnet_blocks = nn.Sequential(
            *[
                ResnetBlock(
                    n_in=n_in if i == 0 else n_hidden,
                    n_out=n_hidden,
                    internal_activation=activation,
                    output_activation=activation,
                )
                for i in range(n_layers)
            ]
        )
        self.fc = Dense(in_features=n_hidden, out_features=n_out)

    @auto_move_data
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.resnet_blocks(inputs)
        return self.fc(h)


class AttentionBlock(nn.Module):
    """Attention block matching the JAX/Flax MultiHeadDotProductAttention architecture.

    Implements the same computation as the Flax version:

    1. Project query/kv to (outerprod_dim, 1) via DenseGeneral-equivalent.
    2. Multi-head attention with internal Q/K/V projections from dim 1.
    3. Output projection to n_channels features (NOT n_channels * n_heads).
    4. MLP processing on flattened attention output (outerprod_dim * n_channels).

    Parameters
    ----------
    query_dim
        Dimension of the query input.
    kv_dim
        Dimension of the kv input.
    out_dim
        Dimension of the output.
    outerprod_dim
        Dimension of the outer product.
    n_channels
        Number of channels (output features from attention).
    n_heads
        Number of heads.
    dropout_rate
        Dropout rate.
    n_hidden_mlp
        Number of hidden units in the MLP.
    n_layers_mlp
        Number of layers in the MLP.
    stop_gradients_mlp
        Whether to stop gradients through the MLP.
    activation
        Activation function to use.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        out_dim: int,
        outerprod_dim: int = 16,
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        n_hidden_mlp: int = 32,
        n_layers_mlp: int = 1,
        stop_gradients_mlp: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = _gelu,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.out_dim = out_dim
        self.outerprod_dim = outerprod_dim
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.n_hidden_mlp = n_hidden_mlp
        self.n_layers_mlp = n_layers_mlp
        self.stop_gradients_mlp = stop_gradients_mlp
        self.activation = activation

        # Projection to (outerprod_dim, 1) — matches JAX DenseGeneral((outerprod_dim, 1))
        self.query_proj = nn.Linear(in_features=query_dim, out_features=outerprod_dim, bias=False)
        self.kv_proj = nn.Linear(in_features=kv_dim, out_features=outerprod_dim, bias=False)

        # Q/K/V projections matching Flax MultiHeadDotProductAttention.
        # depth_per_head = n_channels (= qkv_features // n_heads where qkv_features = n_channels * n_heads)
        self.depth_per_head = n_channels
        qkv_dim = n_heads * self.depth_per_head
        self.q_proj = nn.Linear(in_features=1, out_features=qkv_dim, bias=True)
        self.k_proj = nn.Linear(in_features=1, out_features=qkv_dim, bias=True)
        self.v_proj = nn.Linear(in_features=1, out_features=qkv_dim, bias=True)

        # Output projection: (n_heads * depth_per_head) → n_channels
        self.out_proj = nn.Linear(in_features=qkv_dim, out_features=n_channels, bias=True)

        # MLP input dim: outerprod_dim * n_channels (NOT * n_heads)
        self.mlp_eps = MLP(
            n_in=outerprod_dim * n_channels,
            n_out=outerprod_dim,
            n_hidden=n_hidden_mlp,
            n_layers=n_layers_mlp,
            activation=activation,
        )
        self.mlp_residual = MLP(
            n_in=outerprod_dim + query_dim,
            n_out=out_dim,
            n_hidden=n_hidden_mlp,
            n_layers=n_layers_mlp,
            activation=activation,
        )

    @auto_move_data
    def forward(
        self,
        query_embed: torch.Tensor,
        kv_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        query_embed
            Query embeddings of shape ``(batch, query_dim)`` or
            ``(mc_samples, batch, query_dim)``.
        kv_embed
            Key-value embeddings of shape ``(batch, kv_dim)`` or
            ``(mc_samples, batch, kv_dim)``.

        Returns
        -------
        Residual tensor of shape ``(batch, out_dim)`` or ``(mc_samples, batch, out_dim)``.
        """
        has_mc_samples = query_embed.ndim == 3

        if self.stop_gradients_mlp:
            query_embed_stop = query_embed.detach()
        else:
            query_embed_stop = query_embed

        # Project to (*, outerprod_dim, 1)
        query_for_att = self.query_proj(query_embed_stop).unsqueeze(-1)
        kv_for_att = self.kv_proj(kv_embed).unsqueeze(-1)

        # Flatten mc_samples × batch for attention
        if has_mc_samples:
            mc_samples, batch_size = query_embed.shape[0], query_embed.shape[1]
            query_embed_flat = query_embed.reshape(mc_samples * batch_size, -1)
            query_for_att = query_for_att.reshape(mc_samples * batch_size, self.outerprod_dim, 1)
            kv_for_att = kv_for_att.reshape(mc_samples * batch_size, self.outerprod_dim, 1)
        else:
            query_embed_flat = query_embed

        # Q/K/V projections: (batch, outerprod_dim, 1) → (batch, outerprod_dim, qkv_dim)
        q = self.q_proj(query_for_att)
        k = self.k_proj(kv_for_att)
        v = self.v_proj(kv_for_att)

        # Reshape for multi-head attention:
        # (batch, outerprod_dim, n_heads * depth) → (batch, n_heads, outerprod_dim, depth)
        flat_batch = q.shape[0]

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(
                flat_batch, self.outerprod_dim, self.n_heads, self.depth_per_head
            ).transpose(1, 2)

        q, k, v = _to_heads(q), _to_heads(k), _to_heads(v)

        # Scaled dot-product attention
        dropout_p = self.dropout_rate if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        # (batch, n_heads, outerprod_dim, depth_per_head)

        # Transpose back and flatten heads: → (batch, outerprod_dim, n_heads * depth_per_head)
        attn_out = attn_out.transpose(1, 2).reshape(
            flat_batch, self.outerprod_dim, self.n_heads * self.depth_per_head
        )

        # Output projection: → (batch, outerprod_dim, n_channels)
        eps = self.out_proj(attn_out)

        # Flatten to (batch, outerprod_dim * n_channels)
        eps = eps.reshape(flat_batch, self.outerprod_dim * self.n_channels)

        eps_ = self.mlp_eps(eps)
        inputs = torch.cat([query_embed_flat, eps_], dim=-1)
        residual = self.mlp_residual(inputs)

        if has_mc_samples:
            residual = residual.reshape(mc_samples, batch_size, -1)

        return residual


class EncoderUZ(nn.Module):
    """Attention-based encoder from ``u`` to ``z``.

    Decomposes the cell-level base representation ``u`` into a sample-aware
    residual ``eps`` via attention over a per-sample embedding table, giving
    ``z = z_base + eps``.

    When ``n_latent_u is None`` (isomorphic dims), ``z_base = u`` directly
    (no projection), keeping the decoder input dimension unchanged. This is
    the expected setting for TotalVI / MultiVI integration.

    Parameters
    ----------
    n_latent
        Number of latent variables (output dim = z dim).
    n_sample
        Number of samples/donors.
    n_latent_u
        Number of latent variables for ``u``. If ``None``, isomorphic dims
        are assumed (``n_latent_u == n_latent``) and no projection is applied.
    n_latent_sample
        Dimension of the per-sample embedding.
    n_channels
        Number of channels in the attention block.
    n_heads
        Number of heads in the attention block.
    dropout_rate
        Dropout rate for attention.
    stop_gradients
        Whether to stop gradients through ``u`` before attention.
    stop_gradients_mlp
        Whether to stop gradients through the MLP in the attention block.
    use_map
        If ``True``, use the MAP (deterministic) estimate: attention produces one
        residual directly. If ``False``, produces mean + log-var of a Normal.
        Must be ``True`` when using ``kl_z = -log p(eps)`` as the KL term.
    n_hidden
        Number of hidden units in the MLP.
    n_layers
        Number of layers in the MLP.
    activation
        Activation function for the MLP.
    """

    def __init__(
        self,
        n_latent: int,
        n_sample: int,
        n_latent_u: int | None = None,
        n_latent_sample: int = 16,
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        stop_gradients: bool = False,
        stop_gradients_mlp: bool = False,
        use_map: bool = True,
        n_hidden: int = 32,
        n_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = _gelu,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_sample = n_sample
        self.n_latent_u = n_latent_u if n_latent_u is not None else n_latent
        self.n_latent_sample = n_latent_sample
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.stop_gradients = stop_gradients
        self.stop_gradients_mlp = stop_gradients_mlp
        self.use_map = use_map
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation

        self.layer_norm = nn.LayerNorm(self.n_latent_u, eps=1e-6)
        self.embedding = nn.Embedding(self.n_sample, self.n_latent_sample)
        # Initialize with same std as JAX version
        init.normal_(self.embedding.weight, std=0.1)
        self.layer_norm_embed = nn.LayerNorm(self.n_latent_sample, eps=1e-6)

        n_outs = 1 if self.use_map else 2
        self.attention_block = AttentionBlock(
            query_dim=self.n_latent_u,
            kv_dim=self.n_latent_sample,
            out_dim=n_outs * self.n_latent,
            outerprod_dim=self.n_latent_sample,
            n_channels=self.n_channels,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            stop_gradients_mlp=self.stop_gradients_mlp,
            n_hidden_mlp=self.n_hidden,
            n_layers_mlp=self.n_layers,
            activation=self.activation,
        )

        # Projection from n_latent_u → n_latent only when dims are NOT isomorphic
        if n_latent_u is not None:
            self.fc = nn.Linear(self.n_latent_u, self.n_latent)
        else:
            self.fc = None  # isomorphic: z_base = u directly

    @auto_move_data
    def forward(
        self,
        u: torch.Tensor,
        sample_covariate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute z_base and eps from u and sample covariate.

        Parameters
        ----------
        u
            Base cell representation of shape ``(batch, n_latent_u)`` or
            ``(mc_samples, batch, n_latent_u)``.
        sample_covariate
            Integer sample index of shape ``(batch,)`` or ``(batch, 1)``.

        Returns
        -------
        z_base : torch.Tensor
            Base z before residual: shape ``(batch, n_latent)`` or
            ``(mc_samples, batch, n_latent)``.
        eps : torch.Tensor
            Sample-specific residual: same shape as ``z_base``.
        """
        sample_covariate = sample_covariate.to(torch.int64).flatten()
        has_mc_samples = u.ndim == 3
        u_stop = u if not self.stop_gradients else u.detach()
        u_ = self.layer_norm(u_stop)

        sample_embed = self.layer_norm_embed(self.embedding(sample_covariate))

        if has_mc_samples:
            sample_embed = sample_embed.unsqueeze(0).expand(u_.shape[0], -1, -1)

        residual = self.attention_block(query_embed=u_, kv_embed=sample_embed)

        if self.fc is not None:
            z_base = self.fc(u_stop)
            return z_base, residual
        else:
            # Isomorphic: z_base = u (preserves decoder input unchanged)
            return u, residual
