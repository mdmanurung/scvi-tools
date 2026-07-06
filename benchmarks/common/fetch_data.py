"""Download and validate benchmark datasets (Figshare).

Figshare often returns HTTP 202 while a file is being prepared. Run from a networked shell; if
downloads fail here, fetch manually and re-run ``--validate-only``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

# Vignette / tutorial assets (smoke tests)
VIGNETTE = {
    "Roider_et_al_BNHL_panel1.h5ad": ("56891468", 500_000),
    "Roider_et_al_BNHL_panel2.h5ad": ("56891471", 400_000),
    "Nunez_PBMCs_batch1.fcs": ("55982654", 1_000_000),
    "Nunez_PBMCs_batch2.fcs": ("55982657", 1_000_000),
}

# Full-cohort ingest targets (issue cytovi-benchmark/01–02) — paths are conventions, not auto URLs
FULL_COHORT_NOTES = {
    "roider_full": {
        "issue": "cytovi-benchmark/02",
        "source": "https://doi.org/10.6084/m9.figshare.24915633",
        "local_archive": "data/24915633.zip",
        "extract": "python -m benchmarks.common.ingest --extract-roider",
        "deliver": "data/roider_full/merged.h5ad",
        "note": "Raw .fcs per patient; gate + arcsinh(500) + 10k T cells/patient",
    },
    "nunez_full": {
        "issue": "cytovi-benchmark/01",
        "source": "Figshare 55982654 + 55982657 (all cells after QC, not vignette subsample)",
        "local_files": "data/Nunez_PBMCs_batch{1,2}.fcs",
        "deliver": "benchmarks/cytovi/data/nunez/",
        "note": "arcsinh(2000) + labels; exclude ambiguous cells for integration",
    },
    "kreutmair": {
        "issue": "cytovi-benchmark A-D4",
        "source": "https://doi.org/10.17632/ffkvft27ds.2",
        "local_archive": "data/ffkvft27ds-2.zip",
        "extract": "python -m benchmarks.common.ingest --extract-kreutmair",
        "deliver": "data/kreutmair/",
    },
}


def _download_one(name: str, figshare_id: str, dest: Path, retries: int, wait: float) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://figshare.com/ndownloader/files/{figshare_id}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if data:
                        dest.write_bytes(data)
                        return True
                print(f"  {name}: HTTP {resp.status} (attempt {attempt}/{retries})")
        except OSError as exc:
            print(f"  {name}: {type(exc).__name__}: {exc} (attempt {attempt}/{retries})")
        time.sleep(wait)
    return False


def validate_vignette(data_dir: Path) -> dict:
    """Return per-file status: present, size, passes minimum size heuristic."""
    out = {}
    for name, (_, min_bytes) in VIGNETTE.items():
        path = data_dir / name
        size = path.stat().st_size if path.exists() else 0
        out[name] = {
            "path": str(path),
            "size_bytes": size,
            "ok": size >= min_bytes,
        }
    return out


def fetch_vignette(data_dir: Path, retries: int = 8, wait: float = 15.0) -> dict:
    """Fetch vignette assets from Figshare and return validation status."""
    results = {}
    for name, (fig_id, min_bytes) in VIGNETTE.items():
        dest = data_dir / name
        ok = _download_one(name, fig_id, dest, retries=retries, wait=wait)
        size = dest.stat().st_size if dest.exists() else 0
        results[name] = {"downloaded": ok, "size_bytes": size, "ok": size >= min_bytes}
    return results


def main():
    """Run the benchmark data fetch/validation CLI."""
    ap = argparse.ArgumentParser(description="Fetch or validate CytoVI/CytoANVI vignette data")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/cytoanvi/data"),
        help="Directory for vignette files (default: benchmarks/cytoanvi/data)",
    )
    ap.add_argument("--fetch", action="store_true", help="Attempt Figshare download")
    ap.add_argument("--validate-only", action="store_true", help="Check files on disk only")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--wait", type=float, default=15.0, help="Seconds between Figshare retries")
    ap.add_argument(
        "--list-full-cohort",
        action="store_true",
        help="Print full-cohort ingest notes",
    )
    args = ap.parse_args()

    if args.list_full_cohort:
        print(json.dumps(FULL_COHORT_NOTES, indent=2))
        return

    data_dir = args.data_dir
    if args.fetch:
        print(f"Fetching vignette assets into {data_dir} ...")
        results = fetch_vignette(data_dir, retries=args.retries, wait=args.wait)
    else:
        results = validate_vignette(data_dir)

    report = {
        "data_dir": str(data_dir),
        "vignette": results,
        "all_ok": all(v["ok"] for v in results.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["all_ok"]:
        print(
            "\nManual fetch (from a networked machine):\n"
            "  python -m benchmarks.common.fetch_data --list-full-cohort\n"
            "  See benchmarks/cytoanvi/README.md for curl one-liners."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
