# Bioinformatics QC Checklist

Use before finalizing any benchmark run or analysis for publication.

## Data QC

- [ ] All datasets listed in `data/DATA_MANIFEST.md` with correct status (raw/processed/derived)
- [ ] Batch column present and has >1 unique value (required for scib metrics)
- [ ] Each batch has ≥10 cells
- [ ] `nan_layer` (or equivalent missing-feature mask) is correct for multi-panel datasets
- [ ] Raw and processed data are in separate paths; raw files untouched

## Benchmark QC

- [ ] Benchmarks run at full `max_epochs` (not vignette-scale)
- [ ] Full cohort data used (not subsampled)
- [ ] ≥3 random seeds per task
- [ ] Results aggregated with `aggregate_results.py`; mean ± SD reported
- [ ] Baseline receives identical data splits as comparison model
- [ ] Baseline API method documented in `benchmarks/ANALYSIS_MANIFEST.md`

## Model QC

- [ ] Round-trip test: save model → load from path → inference produces identical output
- [ ] Surgery from saved path tested (not just from in-memory model)
- [ ] Classifier tested with all-unlabeled input (n_labels=0 edge case)
- [ ] Fisher/EWC computation logs subsample fraction

## Publication gate (CytoANVI)

- [ ] B1 macro-F1 ≥ CytoVI k-NN + 0.03 (mean, ≥3 seeds)
- [ ] B2 scib bio ≥ CytoVI, batch within 0.05
- [ ] B3 concordance ≥ 0.80
- [ ] B4 continual F1 within 0.03 of fresh retrain
- [ ] B5 AUROC ≥ 0.70 on held-out types
- [ ] B6 λ sweep complete, default documented
- [ ] B8 HCE ≥ flat CE
- [ ] B9 mapQC passing on controls
