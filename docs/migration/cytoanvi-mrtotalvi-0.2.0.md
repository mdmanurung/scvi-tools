# Migrating to cytoanvi 0.2.0

Version 0.2.0 intentionally fails closed where 0.1.0 accepted ambiguous or scientifically unsafe
states. See the [authoritative usage-readiness matrix](../usage_readiness.md) before using a
capability.

## CytoANVI

| 0.1.0 behavior | 0.2.0 contract | Required migration |
| --- | --- | --- |
| Non-null `adversarial_classifier` was ignored | Every non-null value raises before trainer construction | Remove it; no adversarial objective is available |
| Reloaded continual state could train without replay | Active continual state without replay raises | Reconstruct replay with `load_query_data_with_replay(..., replay_adata=..., control_adata=...)` |
| Stable TTA methods returned novelty-like scores | Stable and indirect TTA novelty entry points refuse use | Treat prior TTA evidence as negative; use only explicitly experimental code for method development |
| One TTA mask could be shared by a batch | Experimental masks are cell-specific and seeded | Pass a deterministic seed and retain cell order for reproducibility |
| Invalid calibration could yield `NaN` threshold | Empty or non-finite calibration raises | Validate calibration data before experimental evaluation |
| Empirical priors/weights used all registered labels | They use the realized training indices only | Do not treat validation labels as training authority |
| mapQC private patch accepted an unpinned version | Exactly mapQC 0.1.1 is accepted | Install/use 0.1.1 under separately authorized dependency management |

`control_adata` is required for the implemented EWC-plus-replay continual objective. No public
EWC-only compatibility mode is introduced.

## MrTotalVI prior migration

`u_prior` is exactly `standard`, `mog`, or `vamp`; new calls default to `mog`.
`u_prior_mixture` is deprecated and is interpreted only as follows:

| `u_prior` | `u_prior_mixture` | Resolved prior |
| --- | --- | --- |
| `standard` | `None` | `standard` |
| `mog` | `None` | `mog` |
| `vamp` | `None` | `vamp` |
| `standard` | `False` | `standard` plus deprecation warning |
| `mog` | `True` | `mog` plus deprecation warning |
| `vamp` | `True` | `vamp` plus deprecation warning |

Every other combination and every unknown enum raises before module construction. A historical
checkpoint with a contradictory combination is unsupported. A consistent historical checkpoint
loads with its objective intact and is resaved with explicit resolved metadata.

## MrTotalVI supervision and data contracts

| 0.1.0 behavior | 0.2.0 contract | Required migration |
| --- | --- | --- |
| Registered labels could silently supervise a mixture prior | New calls default to `u_prior_supervision="none"`, weight `0.0` | Set `u_prior_supervision="labels"` and a finite positive weight only when intentional |
| Count checks sampled values and warned | RNA and protein values are exhaustively checked before setup mutation | Supply finite, non-negative, integer-like raw counts |
| Unknown/duplicate subsets or ambiguous metadata could be accepted | Sample/donor/covariate mappings and ordered subsets are strict | Repair metadata and provide unique known samples in intended order |
| Legacy DE returned inferential-looking p-values/LFC | Public DE refuses both legacy and centered-v2 calls | Use donor-pseudobulk PyDESeq2, edgeR, or dreamlet for biological inference |
| `use_vmap=True` was ignored | It raises before inference/statistics | Use the explicit loop path (`False`) |
| Multi-file inputs checked only protein width | Protein names must be non-empty, unique, and identical in order | Harmonize authoritative protein axes before loading |
| `MrTotalVIBatchDataModule` implied streaming training | It is absent from the stable export | Treat any private adapter as registry plumbing, not a training workflow |

Model summaries and checkpoints record hierarchy mode, encoder mode, resolved prior, supervision
mode, and supervision weight. `get_latent_representation(give_z=True)` has width `n_latent`;
`give_z=False` has width `n_latent_u`.
