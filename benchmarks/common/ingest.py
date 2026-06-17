"""Extract and inventory full-cohort benchmark archives dropped in repo ``data/``."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_DATA = _REPO_ROOT / "data"

ROIDER_ZIP = "24915633.zip"
KREUTMAIR_ZIP = "ffkvft27ds-2.zip"
ROIDER_NESTED = re.compile(r"FlowCytometryData_Part[A-D].*\.zip$", re.I)
ROIDER_FCS = re.compile(
    r"Panel(?P<panel>[12])_.*_(?P<sample>LN[A-Za-z0-9]+)(?:_\d+)?\.fcs$",
    re.I,
)
KREUTMAIR_FCS = re.compile(r"(?P<panel>Myeloid|Lymphoid) panel/(?P<sample>\d+)_clean\.fcs$", re.I)


def _repo_data() -> Path:
    return _REPO_DATA


def _extract_zip(zip_path: Path, dest: Path, *, members_filter=None) -> int:
    """Extract ``zip_path`` into ``dest``; return number of files written."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or "__MACOSX" in info.filename or info.filename.endswith("/.DS_Store"):
                continue
            if members_filter and not members_filter(info.filename):
                continue
            target = dest / info.filename
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def extract_roider(data_dir: Path | None = None) -> Path:
    """Unzip Figshare 24915633 (outer + nested Part A–D) into ``data/roider_raw/``."""
    data_dir = data_dir or _repo_data()
    outer = data_dir / ROIDER_ZIP
    if not outer.exists():
        raise FileNotFoundError(f"Missing {outer}")
    root = data_dir / "roider_raw"
    _extract_zip(outer, root)
    for nested in sorted(root.glob("FlowCytometryData_Part*.zip")):
        _extract_zip(nested, root / nested.stem)
    return root


def extract_kreutmair(data_dir: Path | None = None) -> Path:
    """Unzip Mendeley ffkvft27ds.2 into ``data/kreutmair/``."""
    data_dir = data_dir or _repo_data()
    outer = data_dir / KREUTMAIR_ZIP
    if not outer.exists():
        raise FileNotFoundError(f"Missing {outer}")
    root = data_dir / "kreutmair"
    _extract_zip(outer, root)
    return root


def _walk_fcs(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.fcs") if "__MACOSX" not in str(p) and "Compensation" not in p.name]


def inventory_roider(root: Path) -> dict:
    """Summarize Panel1/Panel2 FCS files under ``roider_raw``."""
    panel_counts: Counter[str] = Counter()
    patients: set[str] = set()
    examples: list[str] = []
    for path in _walk_fcs(root):
        m = ROIDER_FCS.search(path.name)
        if not m:
            continue
        panel_counts[f"panel{m.group('panel')}"] += 1
        patients.add(m.group("sample"))
        if len(examples) < 8:
            examples.append(str(path.relative_to(root)))
    return {
        "root": str(root),
        "n_fcs_panel": dict(panel_counts),
        "n_patients": len(patients),
        "patient_ids_sample": sorted(patients)[:20],
        "fcs_examples": examples,
    }


def inventory_kreutmair(root: Path) -> dict:
    panel_counts: Counter[str] = Counter()
    samples: set[str] = set()
    for path in _walk_fcs(root):
        rel = str(path.relative_to(root))
        m = KREUTMAIR_FCS.search(rel.replace("\\", "/"))
        if m:
            panel_counts[m.group("panel")] += 1
            samples.add(m.group("sample"))
    xlsx = sorted(str(p.relative_to(root)) for p in root.rglob("Mendeley_*.xlsx"))
    return {
        "root": str(root),
        "n_fcs_by_panel": dict(panel_counts),
        "n_samples": len(samples),
        "metadata_xlsx": xlsx,
    }


def write_inventory(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or _repo_data()
    report: dict = {"data_dir": str(data_dir)}
    roider_root = data_dir / "roider_raw"
    if roider_root.exists():
        inv = inventory_roider(roider_root)
        report["roider"] = inv
        (roider_root / "inventory.json").write_text(json.dumps(inv, indent=2))
    k_root = data_dir / "kreutmair"
    if k_root.exists():
        inv = inventory_kreutmair(k_root)
        report["kreutmair"] = inv
        (k_root / "inventory.json").write_text(json.dumps(inv, indent=2))
    out = data_dir / "ingest_inventory.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Extract and inventory full-cohort archives in data/")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--extract-roider", action="store_true")
    ap.add_argument("--extract-kreutmair", action="store_true")
    ap.add_argument("--inventory-only", action="store_true", help="Scan extracted trees only")
    args = ap.parse_args()
    data_dir = args.data_dir or _repo_data()

    if args.extract_roider:
        print(f"Extracting Roider archive -> {data_dir / 'roider_raw'}")
        extract_roider(data_dir)
    if args.extract_kreutmair:
        print(f"Extracting Kreutmair archive -> {data_dir / 'kreutmair'}")
        extract_kreutmair(data_dir)
    if args.inventory_only or args.extract_roider or args.extract_kreutmair:
        report = write_inventory(data_dir)
        print(json.dumps(report, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
