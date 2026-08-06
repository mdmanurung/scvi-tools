# MrTotalVI v2 bounded engineering smoke

`run_bounded_human_comparator.py` exercises package semantics on realistic tensor
shapes. It is deliberately not a scientific analysis:

- the input H5AD and panel map must match their frozen SHA-256 digests;
- 25 cells are selected by cell-ID hash from each of the 20 W00/W22
  `donor_timepoint` groups;
- the fixture retains only counts, `donor_timepoint`, `batch`, and the 130
  biological proteins identified by the frozen panel map;
- the script never reads or filters on `pass_qc`;
- one CPU epoch and bounded counterfactual calls are engineering evidence only.

Every successful invocation creates a new immutable-by-convention run directory
under `.scratch/mrtotalvi-v2/engineering-runs/`. It refuses an existing
destination and never creates a `latest` pointer.
