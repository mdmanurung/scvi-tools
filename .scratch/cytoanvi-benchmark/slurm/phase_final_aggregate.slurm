#!/usr/bin/env bash
# Final publication aggregation: runs ONLY if all 3 B3 seeds SUCCEED (afterok), flips the B3
# manifest entries to complete, then aggregates the full required set (B1,B2,B3,B5,B8) into
# publication_summary.json. afterok (not afterany) so a B3 failure leaves this unsatisfied and
# I handle it manually rather than aggregating a partial set.
#
#SBATCH --job-name=cytoanvi_final_aggregate
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --dependency=afterok:25145151:25145152:25145153
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

MANIFEST="$ROOT/.scratch/cytoanvi-benchmark/publication_manifest.json"
SUMMARY="$RESULTS/publication_summary.json"

echo "=== Final publication aggregation ==="
echo "Start: $(date)"

# B3 seeds succeeded (afterok) and wrote roider_full_b3_s{0,1,2}.json -> mark them complete.
$PY - "$MANIFEST" <<'PYEOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
n = 0
for a in d["artifacts"]:
    if a.get("task") == "b3" and a.get("dataset") == "roider-full":
        a["status"] = "complete"; n += 1
open(p, "w").write(json.dumps(d, indent=2) + "\n")
print(f"flipped {n} B3 manifest entries -> complete")
PYEOF

$PY -m benchmarks.common.aggregate_results --manifest "$MANIFEST" --output "$SUMMARY"

echo "=== publication summary written: $SUMMARY ==="
$PY -c "import json; d=json.load(open('$SUMMARY')); print(json.dumps(d.get('summary', d), indent=2)[:2000])" || true
echo "Done: $(date)"
