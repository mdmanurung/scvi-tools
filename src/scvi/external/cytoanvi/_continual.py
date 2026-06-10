"""Continual-learning machinery for CytoANVI case-control atlas building.

Ports the approach of ``theislab/comparative_atlas`` (``cscanvi``) to CytoANVI and modern
scvi-tools: regularized incremental query updates that anchor to a reference (and healthy
controls) via an Elastic-Weight-Consolidation (EWC) penalty whose Fisher-style parameter
importances are estimated from a replay buffer of reference cells and query controls.

A configured update lives in one place: :class:`ContinualUpdate` owns the reference anchor, both
Fisher importances, the combine rule, and the replay buffer. At training time the query minibatch
flows through the ELBO plus the EWC penalty, and the replay buffer is rehearsed in the ELBO
(Experience Replay), matching the paper's loss.

Query-novelty (test-time-augmentation) uncertainty is a separate concern and lives in
:mod:`~scvi.external.cytoanvi._uncertainty`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from scvi.train import SemiSupervisedTrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData


def zerolike_params_dict(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """``[(name, zeros_like(param))]`` for trainable params (Fisher accumulator init)."""
    return [(k, torch.zeros_like(p)) for k, p in module.named_parameters() if p.requires_grad]


def fisher_importances(model, adata: AnnData) -> list[tuple[str, torch.Tensor]]:
    """Fisher-style parameter importances = mean squared ELBO gradient over ``adata``.

    Estimated on an unfrozen copy of ``model`` so every parameter gets a gradient. Returns CPU
    tensors keyed by parameter name (CPU so they pickle cleanly for save/load; the EWC penalty
    moves them to the live device on use).
    """
    model = deepcopy(model)
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
    return [(k, (v / n_batches).detach().cpu()) for k, v in importances.items()]


class ContinualUpdate:
    """A configured continual case-control update (cscanvi-style EWC + Experience Replay).

    Owns everything the Phase-2 update needs, behind one small interface: the reference **anchor**
    (``old_params``), the reference Fisher (``importances``), the query-control Fisher
    (``ctrl_importances``), the combine rule (``F_reference o F_query_ctrl``), and the **replay
    buffer**. Held by :class:`~scvi.external.cytoanvi.CytoANVAE` as one attribute — present means
    the continual update is active, absent (``None``) means the base path.

    The drift penalty is returned **unscaled**; ``lambda`` (``ewc_importance``) is applied by the
    training plan at train time. The replay buffer is session-scoped and is **not** persisted
    (:meth:`persistable_state` serializes only the anchor + Fishers + combine rule).
    """

    def __init__(
        self,
        old_params: list[tuple[str, torch.Tensor]],
        importances: list[tuple[str, torch.Tensor]],
        ctrl_importances: list[tuple[str, torch.Tensor]] | None = None,
        combine_type: str = "product",
        replay_batches: list[dict[str, torch.Tensor]] | None = None,
    ):
        self.old_params = old_params
        self.importances = importances
        self.ctrl_importances = ctrl_importances
        self.combine_type = combine_type
        self.replay_batches = replay_batches

    @classmethod
    def configure(
        cls,
        reference_model,
        query_model,
        replay_adata: AnnData,
        control_adata: AnnData | None,
        combine_type: str = "product",
    ) -> ContinualUpdate:
        """Build the update at surgery time.

        Snapshots the anchor and reference Fisher from ``reference_model`` (reference shapes),
        the query-control Fisher from ``query_model`` (controls may carry query batches, so their
        importances are computed on the batch-extended query model), and materializes the replay
        buffer via the query model's loader.
        """
        old_params = [
            (k, p.detach().cpu().clone()) for k, p in reference_model.module.named_parameters()
        ]
        importances = fisher_importances(reference_model, replay_adata)
        ctrl_importances = (
            fisher_importances(query_model, control_adata) if control_adata is not None else None
        )
        replay_val = query_model._validate_anndata(replay_adata)
        replay_dl = query_model._make_data_loader(adata=replay_val, batch_size=256, shuffle=True)
        replay_batches = [
            {k: v.detach().cpu() for k, v in tensors.items()} for tensors in replay_dl
        ]
        return cls(old_params, importances, ctrl_importances, combine_type, replay_batches)

    def penalty(self, module: torch.nn.Module) -> torch.Tensor:
        """EWC drift penalty ``sum_k w_k (theta_k - theta_k^ref)^2`` against ``module``'s params.

        ``w_k`` is the reference Fisher, combined with the query-control Fisher per
        ``combine_type``. Params resized by surgery (size mismatch vs the anchor) are skipped.
        Unscaled — the training plan multiplies by ``ewc_importance`` (lambda).
        """
        device = module.device
        cur = dict(module.named_parameters())
        imps = dict(self.importances)
        ctrl = dict(self.ctrl_importances) if self.ctrl_importances is not None else None
        penalty = torch.zeros((), device=device)
        for name, saved in self.old_params:
            p = cur.get(name)
            imp = imps.get(name)
            if p is None or imp is None or p.size() != saved.size():
                continue
            saved = saved.to(device)
            w = imp.to(device)
            if ctrl is not None and name in ctrl:
                c = ctrl[name].to(device)
                w = w * c if self.combine_type == "product" else w + c
            penalty = penalty + (w * (p - saved).pow(2)).sum()
        return penalty

    def next_replay_batch(self, batch_idx: int, device) -> dict[str, torch.Tensor] | None:
        """The replay-buffer minibatch for ``batch_idx`` (cycling), moved to ``device``."""
        if not self.replay_batches:
            return None
        rb = self.replay_batches[batch_idx % len(self.replay_batches)]
        return {k: v.to(device) for k, v in rb.items()}

    def persistable_state(self) -> dict:
        """The save/load-persisted state: anchor + both Fishers + combine rule (no replay buffer).

        The replay buffer (a slice of reference cells) is session-scoped; resuming continual
        *training* after a reload requires re-supplying ``replay_adata``, while
        ``predict`` / ``get_latent_representation`` / ``get_uncertainty`` work immediately.
        """
        return {
            "old_params": self.old_params,
            "importances": self.importances,
            "ctrl_importances": self.ctrl_importances,
            "combine_type": self.combine_type,
        }

    @classmethod
    def from_persistable_state(cls, state: dict) -> ContinualUpdate:
        """Rebuild from :meth:`persistable_state` (replay buffer left empty)."""
        return cls(
            old_params=state["old_params"],
            importances=state["importances"],
            ctrl_importances=state.get("ctrl_importances"),
            combine_type=state.get("combine_type", "product"),
            replay_batches=None,
        )


class CytoANVIContinualTrainingPlan(SemiSupervisedTrainingPlan):
    """Semi-supervised training plan for continual case-control updates (paper-faithful).

    Implements the paper's loss ``L(theta_query) = ELBO(x_query, x_replay) + (lambda/2) F
    (theta_query - theta_ref)^2``:

    - the query minibatch flows through ``module._replay_forward`` (ELBO + the EWC penalty,
      weighted by ``ewc_importance`` = lambda), and
    - a replay-buffer minibatch (reference cells owned by the module's :class:`ContinualUpdate`,
      set by ``CytoANVI.load_query_data_with_replay``) is rehearsed each step by adding its plain
      ELBO (Experience Replay). The replay batches cycle by ``batch_idx``.
    """

    def __init__(self, module, n_classes: int, *, ewc_importance: float = 1.0, **kwargs):
        super().__init__(module, n_classes, **kwargs)
        self.loss_kwargs.update({"ewc_importance": ewc_importance})

    def forward(self, *args, **kwargs):
        """Route the forward pass through the module's replay/EWC forward (ELBO + EWC penalty)."""
        return self.module._replay_forward(*args, **kwargs)

    def _next_replay_batch(self, batch_idx: int):
        """Cycle through the continual update's replay-buffer minibatches, on the module device."""
        cont = getattr(self.module, "continual", None)
        if cont is None:
            return None
        return cont.next_replay_batch(batch_idx, self.module.device)

    def training_step(self, batch, batch_idx):
        """Query ELBO + EWC penalty, plus the rehearsed replay-buffer ELBO."""
        if len(batch) == 2:
            full_dataset, labelled_dataset = batch[0], batch[1]
        else:
            full_dataset, labelled_dataset = batch, None

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        input_kwargs = {"labelled_tensors": labelled_dataset}
        input_kwargs.update(self.loss_kwargs)

        # query minibatch: ELBO + EWC penalty
        _, _, loss_output = self.module._replay_forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss

        # experience replay: add the plain ELBO of a replay-buffer minibatch (no EWC)
        replay_batch = self._next_replay_batch(batch_idx)
        if replay_batch is not None:
            _, _, replay_out = self.module(replay_batch, loss_kwargs={"kl_weight": self.kl_weight})
            loss = loss + replay_out.loss

        self.log("train_loss", loss, on_epoch=True, batch_size=loss_output.n_obs_minibatch)
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        return loss
