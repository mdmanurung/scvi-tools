# Source-review evidence anchor

This tracked file preserves the capability-level conclusions that justified the
usage-readiness remediation packet. It is an evidence anchor, not a replacement
for terminal source, wheel, installed-artifact, or scientific validation.

## Source identity

- Reviewed source commit: `297769d3c62b9228244a05469dc8349a55e4174c`
- Original local review path:
  `.living/outputs/reviews/2026-08-07-cytoanvi-mrtotalvi-usage-readiness.md`
- Original review SHA-256:
  `6709acafbc79152e891db659f543fd429147c110e68e3ab39bdee4be10b3f9db`
- Review date: 2026-08-07
- Review scope: CytoANVI and MrTotalVI source, tests, documentation,
  decisions, benchmark registries, and surviving local benchmark artifacts;
  MrMultiVI readiness was excluded.

The original review is local living-repository state and is intentionally not
part of the usage-readiness source commit. This root-owned anchor makes every
matrix evidence link reconstructable without staging or modifying `.living/`.

## Capability conclusions retained from the review

| Capability | Retained evidence | Readiness boundary |
| --- | --- | --- |
| CytoANVI core transfer | Three-seed Roider macro-F1 was `0.9317 ± 0.0022`, below the reported XGBoost comparator (`0.9516`), with no independent biological truth labels. | Conditional expert use only, with an independently annotated target holdout. |
| CytoANVI same-panel and panel-divergent mapping | Mapping behavior was technically covered, but concordance was not target-label accuracy and divergent panels require a discriminative shared backbone. | Dataset-specific validation is mandatory; no general promotion. |
| CytoANVI hierarchy and integration | Evidence came from narrow or essentially leaf-only hierarchy cases and one main cohort. | Exploratory or conditional only; no universal hierarchy or batch-removal claim. |
| CytoANVI TTA novelty | Three-seed mean OOD AUROC was `0.484 ± 0.005`, while a latent-kNN diagnostic reached `0.906 ± 0.003`. | Stable TTA novelty/thresholding is a no-go. |
| CytoANVI continual update | EWC state persisted but replay cells did not, and no portable lambda or decisive real case/control validation existed. | Experimental only; consequential continual updates are a no-go. |
| CytoANVI mapQC | The reviewed benchmark rejected all Nuñez queries despite strong transfer, and the adapter depended on private upstream behavior. | No-go as an automatic acceptance gate. |
| MrTotalVI core and embeddings | In-memory training and factual/sample-aware latent work were usable, but legacy `u` was sample-conditioned rather than donor-neutral. | Conditional exploratory use only. |
| MrTotalVI prior and supervision | MoG versus Vamp was scientifically unsettled, and registering labels could silently alter the historical objective. | Prior choice must be prespecified; supervision must be explicit. |
| MrTotalVI legacy DA | Three seeds produced `+2.74`, `-9.04`, and `+9.67`, including a sign reversal. | No-go for single-fit decisions or inference. |
| MrTotalVI legacy DE | All 12 tested cell types were anti-concordant with donor-pseudobulk direction (Spearman rho from `-0.126` to `-0.008`). | No-go for biological inference; use replicate-aware pseudobulk methods. |
| MrTotalVI centered-v2 | The estimands were descriptive, registered-sample-only, non-causal, and lacked complete scientific validation. | Engineering preview / exploratory only. |
| MrTotalVI streaming and new-sample inference | No end-to-end streaming training contract or defensible new-sample projection contract existed. | No-go public model-training/inference surfaces. |

## Interpretation rule

These historical findings may support a negative or restricted readiness state.
They cannot promote a capability, substitute for a frozen protocol, provide a
human approval, or bind results to the 0.2.0 candidate artifact. Promotion still
requires the exact terminal evidence and signatures defined by the frozen plan.
