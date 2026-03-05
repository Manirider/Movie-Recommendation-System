#!/usr/bin/env python

import argparse
import os
import sys
import urllib.request
import zipfile

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED_FILES = ["ratings.dat", "movies.dat", "users.dat"]


def download_and_extract(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)

    existing = [f for f in REQUIRED_FILES if os.path.isfile(
        os.path.join(dest_dir, f))]
    if len(existing) == len(REQUIRED_FILES):
        print(f"All files already present in {dest_dir}/ — skipping download.")
        return

    zip_path = os.path.join(dest_dir, "ml-1m.zip")

    print(f"Downloading MovieLens 1M from {MOVIELENS_URL} ...")
    try:
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path, _progress_hook)
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("\nDownload complete.")

    print("Extracting files ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if basename in REQUIRED_FILES:
                with zf.open(member) as src, open(os.path.join(dest_dir, basename), "wb") as dst:
                    dst.write(src.read())
                print(f"  ✓ {basename}")

    os.remove(zip_path)
    print(f"\nDone — dataset ready in {dest_dir}/")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        print(
            f"\r  [{bar}] {pct:3d}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MB)", end="")
    else:
        print(f"\r  {downloaded // 1_048_576} MB downloaded ...", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MovieLens 1M dataset")
    parser.add_argument(
        "--dest", default="data",
        help="Destination directory for .dat files (default: data/)",
    )
    args = parser.parse_args()
    download_and_extract(args.dest)
