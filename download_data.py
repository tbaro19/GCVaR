#!/usr/bin/env python3
"""Automated Data Acquisition Script for GCVaR Benchmark Datasets.

Downloads and extracts public domain benchmark datasets from the UCI Machine Learning Repository:
1. Bike Sharing Dataset (UCI ID 275) -> data/bike/hour.csv
2. Power Consumption of Tetouan City (UCI ID 849) -> data/tetouan/Tetuan City power consumption.csv
3. Electricity Load Diagrams 2011-2014 (UCI ID 321) -> data/electricity/LD2011_2014.txt

Usage:
    python download_data.py              # Download all datasets
    python download_data.py --dataset bike
    python download_data.py --dataset tetouan
"""

from __future__ import annotations
import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"

DATASET_CONFIGS = {
    "bike": {
        "name": "UCI Bike Sharing Dataset (ID 275)",
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "target_dir": DATA_DIR / "bike",
        "target_file": "hour.csv",
        "min_bytes": 1_000_000,
    },
    "tetouan": {
        "name": "Power Consumption of Tetouan City (ID 849)",
        "url": "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip",
        "target_dir": DATA_DIR / "tetouan",
        "target_file": "Tetuan City power consumption.csv",
        "min_bytes": 4_000_000,
    },
    "electricity": {
        "name": "Electricity Load Diagrams 2011-2014 (ID 321)",
        "url": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "target_dir": DATA_DIR / "electricity",
        "target_file": "LD2011_2014.txt",
        "min_bytes": 10_000_000,
    },
}


def download_and_extract(dataset_key: str) -> bool:
    cfg = DATASET_CONFIGS[dataset_key]
    target_dir: Path = cfg["target_dir"]
    target_file = target_dir / cfg["target_file"]

    if target_file.exists() and target_file.stat().st_size >= cfg["min_bytes"]:
        print(f"[{dataset_key.upper()}] Dataset already exists at: {target_file} (skipping)", flush=True)
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{dataset_key.upper()}] Downloading {cfg['name']}...", flush=True)
    print(f"  URL: {cfg['url']}", flush=True)

    try:
        req = urllib.request.Request(
            cfg["url"],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GCVaR-Reproducibility/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read()

        print(f"  Downloaded {len(content) / (1024 * 1024):.2f} MB. Extracting archive...", flush=True)
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                # Match target filename regardless of archive folder hierarchy
                if Path(member).name.lower() == cfg["target_file"].lower():
                    source_bytes = zf.read(member)
                    target_file.write_bytes(source_bytes)
                    print(f"  Extracted: {target_file} ({len(source_bytes) / (1024 * 1024):.2f} MB)", flush=True)
                    return True

        # Fallback: extract entire zip into target directory
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(target_dir)
        print(f"  Extracted all files to {target_dir}", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download {cfg['name']}: {e}", file=sys.stderr, flush=True)
        print(f"  Please download manually from: {cfg['url']}", file=sys.stderr, flush=True)
        print(f"  Extract '{cfg['target_file']}' into '{target_dir}'.", file=sys.stderr, flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download operational datasets for GCVaR benchmarks")
    parser.add_argument("--dataset", choices=["all", "bike", "tetouan", "electricity"], default="all",
                        help="Dataset to download (default: all)")
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    success_count = sum(download_and_extract(k) for k in targets)

    print(f"\nCompleted: {success_count}/{len(targets)} datasets verified.", flush=True)


if __name__ == "__main__":
    main()

