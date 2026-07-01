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
:mod:`~cytoanvi._uncertainty`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from scvi.train import SemiSupervisedTrainingPlan

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


def zerolike_params_dict(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """``[(name, zeros_like(param))]`` for trainable params (Fisher accumulator init)."""
    return [(k, torch.zeros_like(p)) for k, p in module.named_parameters() if p.requires_grad]


def fisher_importances(
    model,
    adata: AnnData,
    *,
    max_cells: int | None = 10_000,
    batch_size: int = 256,
    seed: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    """Fisher-style parameter importances = mean squared ELBO gradient over ``adata``.

    Runs on the live ``model`` without a deepcopy: ``requires_grad`` flags are snapshotted,
    temporarily set to ``True`` for all params, then restored in a ``finally`` block. Grads are
    accumulated into a separate ``importances`` dict and cleared afterward — the live optimizer
    state and parameter *values* are never modified.

    Returns CPU tensors keyed by parameter name (CPU so they pickle cleanly for save/load; the EWC
    penalty moves them to the live device on use).

    Uses the **semi-supervised ELBO** including the classification term for labeled cells; Fisher
    importances thus protect classifier weights proportional to their contribution. For large
    references, subsamples to ``max_cells`` cells before estimation.
    """
    if adata.n_obs == 0:
        raise ValueError("fisher_importances requires a non-empty AnnData")
    adata = model._validate_anndata(adata)
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
    logger.info("Estimating Fisher importances on %d cells.", adata.n_obs)
    scdl = model._make_data_loader(adata=adata, batch_size=batch_size)

    # Snapshot requires_grad so we can restore the caller's freeze state afterward.
    # The snapshot must be outside try so the finally block always has the original flags.
    grad_flags = {name: p.requires_grad for name, p in model.module.named_parameters()}
    n_batches = 0
    try:
        # Mutation, allocation, and eval() are inside try so any failure (OOM etc.) falls
        # through to finally and the caller's requires_grad state is always restored.
        for p in model.module.parameters():
            p.requires_grad = True
        importances = dict(zerolike_params_dict(model.module))
        was_training = model.module.training
        model.module.eval()  # eval() outside the grad context: disables dropout/BN updates
        # enable_grad guards against an outer torch.inference_mode() context; raw (unclipped)
        # gradients are required here — gradient clipping must NOT apply to this pass.
        with torch.enable_grad():
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
    finally:
        # Restore requires_grad flags and clear accumulated grads so the caller's optimizer
        # state is unaffected.  Guard was_training in case the try block failed before it
        # was assigned (e.g. OOM during zerolike_params_dict).
        for name, p in model.module.named_parameters():
            p.requires_grad = grad_flags.get(name, p.requires_grad)
        model.module.zero_grad(set_to_none=True)
        if locals().get("was_training"):
            model.module.train()
    n_batches = max(n_batches, 1)
    return [(k, (v / n_batches).detach().cpu()) for k, v in importances.items()]


class ContinualUpdate:
    """A configured continual case-control update (cscanvi-style EWC + Experience Replay).

    Owns everything the Phase-2 update needs, behind one small interface: the reference **anchor**
    (``old_params``), the reference Fisher (``importances``), the query-control Fisher
    (``ctrl_importances``), the combine rule (``F_reference o F_query_ctrl``), and the **replay
    buffer**. Held by :class:`~cytoanvi.CytoANVAE` as one attribute — present means
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
        combine_type: Literal["product", "additive"] = "product",
        replay_batches: list[dict[str, torch.Tensor]] | None = None,
    ):
        if combine_type not in ("product", "additive"):
            raise ValueError(
                f"combine_type must be 'product' or 'additive'; got {combine_type!r}."
            )
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
        combine_type: Literal["product", "additive"] = "product",
        seed: int = 0,
    ) -> ContinualUpdate:
        """Build the update at surgery time.

        Snapshots the anchor and reference Fisher from ``reference_model`` (reference shapes),
        the query-control Fisher from ``query_model`` (controls may carry query batches, so their
        importances are computed on the batch-extended query model), and materializes the replay
        buffer via the query model's loader.

        ``seed`` controls the Fisher subsampling and replay-buffer ordering so that two calls with
        the same inputs and the same seed produce identical ``importances`` and ``replay_batches``.
        """
        old_params = [
            (k, p.detach().cpu().clone()) for k, p in reference_model.module.named_parameters()
        ]
        importances = fisher_importances(reference_model, replay_adata, seed=seed)
        ctrl_importances = (
            fisher_importances(query_model, control_adata, seed=seed)
            if control_adata is not None
            else None
        )
        replay_val = query_model._validate_anndata(replay_adata)
        # Deterministic ordering via an explicit seeded permutation rather than shuffle=True,
        # which would draw from the global RNG and produce non-reproducible replay buffers.
        perm = np.random.default_rng(seed).permutation(len(replay_val))
        replay_dl = query_model._make_data_loader(
            adata=replay_val, indices=perm, batch_size=256, shuffle=False
        )
        replay_batches = [
            {k: v.detach().cpu() for k, v in tensors.items()} for tensors in replay_dl
        ]
        return cls(old_params, importances, ctrl_importances, combine_type, replay_batches)

    def to_device(self, device) -> None:
        """Pre-load the constant anchor/Fisher tensors onto ``device`` once before training.

        Called from :meth:`CytoANVIContinualTrainingPlan.on_train_start` so that
        :meth:`penalty` reads from an already-resident cache rather than re-`.to(device)`-ing
        constant tensors every training step.  The ``_dev_*`` dicts are excluded from
        :meth:`persistable_state` and from pickling (see ``__getstate__``).
        """
        self._dev_old: dict[str, torch.Tensor] = {k: v.to(device) for k, v in self.old_params}
        self._dev_imps: dict[str, torch.Tensor] = {k: v.to(device) for k, v in self.importances}
        self._dev_ctrl: dict[str, torch.Tensor] | None = (
            {k: v.to(device) for k, v in self.ctrl_importances}
            if self.ctrl_importances is not None
            else None
        )

    def __getstate__(self) -> dict:
        """Exclude device-cached tensors from pickling (they are derived from CPU state)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_dev")}

    def penalty(self, module: torch.nn.Module) -> torch.Tensor:
        """EWC drift penalty ``sum_k w_k (theta_k - theta_k^ref)^2`` against ``module``'s params.

        ``w_k`` is the reference Fisher, combined with the query-control Fisher per
        ``combine_type``. Params resized by surgery (size mismatch vs the anchor) are skipped.
        Unscaled — the training plan multiplies by ``ewc_importance`` (lambda).
        """
        device = module.device
        cur = dict(module.named_parameters())
        # Use device-cached dicts (built by to_device) when available; fall back to per-call
        # .to(device) when penalty is invoked outside training (e.g. evaluation, Fisher pass).
        # NOTE: use `is not None` rather than truthiness — empty dicts are falsy but valid.
        _cached_old = getattr(self, "_dev_old", None)
        old: dict[str, torch.Tensor] = (
            _cached_old if _cached_old is not None else dict(self.old_params)
        )
        _cached_imps = getattr(self, "_dev_imps", None)
        imps: dict[str, torch.Tensor] = (
            _cached_imps if _cached_imps is not None else dict(self.importances)
        )
        _cached_ctrl = getattr(self, "_dev_ctrl", None)
        # _dev_ctrl itself can be None when ctrl_importances is None, so we check the attribute
        # existence separately from the None-meaning-"no control Fisher" case.
        ctrl: dict[str, torch.Tensor] | None
        if hasattr(self, "_dev_ctrl"):
            ctrl = _cached_ctrl  # may legitimately be None (no control Fisher)
        else:
            ctrl = dict(self.ctrl_importances) if self.ctrl_importances is not None else None
        penalty = torch.zeros((), device=device)
        for name, saved in old.items():
            p = cur.get(name)
            imp = imps.get(name)
            if p is None or imp is None or p.size() != saved.size():
                continue
            # Fallback .to(device) is a no-op when tensors are already on the target device.
            saved = saved.to(device)
            w = imp.to(device)
            if ctrl is not None and name in ctrl:
                c = ctrl[name].to(device)
                if self.combine_type == "product":
                    w = torch.clamp(w * c, min=1e-10)
                else:
                    w = w + c
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
        combine_type = state.get("combine_type", "product")
        try:
            return cls(
                old_params=state["old_params"],
                importances=state["importances"],
                ctrl_importances=state.get("ctrl_importances"),
                combine_type=combine_type,
                replay_batches=None,
            )
        except ValueError as exc:
            raise ValueError(
                f"Saved model has invalid combine_type {combine_type!r}; was the model saved "
                "with an older pre-validation version?"
            ) from exc


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
        # Manual optimization so gradient clipping is applied inside training_step
        # alongside the composite EWC+replay loss; Trainer(gradient_clip_val=...) is
        # incompatible with manual mode — use plan_kwargs["gradient_clip_norm"] instead.
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        """Route the forward pass through the module's replay/EWC forward (ELBO + EWC penalty)."""
        return self.module._replay_forward(*args, **kwargs)

    def on_train_start(self) -> None:
        """Pre-load Fisher/anchor tensors to the training device once, before the first step."""
        cont = getattr(self.module, "continual", None)
        if cont is not None:
            cont.to_device(self.module.device)

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

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.module.parameters()),
                self.gradient_clip_norm,
            )
        opt.step()

        self.log("train_loss", loss, on_epoch=True, batch_size=loss_output.n_obs_minibatch)
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        return loss

    def on_train_epoch_end(self):
        """Step the LR scheduler at epoch end (manual-optimization mode).

        The early-return guard is correct: the base ``configure_optimizers`` only constructs a
        ``ReduceLROnPlateau`` scheduler, and only when ``reduce_lr_on_plateau=True``.  No other
        scheduler type is ever present for this plan, so calling ``sch.step()`` unconditionally
        would raise on ``None``.
        """
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Step the LR scheduler after validation (manual-optimization mode).

        See ``on_train_epoch_end`` for why the early-return guard is correct.
        """
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])
