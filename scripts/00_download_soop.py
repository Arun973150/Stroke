#!/usr/bin/env python3
"""
Download SOOP Dataset (ds004889) from OpenNeuro to Lambda Labs instance.

Usage:
    python scripts/00_download_soop.py [--data-dir /path/to/data]

Default data directory: /home/ubuntu/data/SOOP_Dataset
"""

import argparse
import os
import subprocess
import sys


def run(cmd, check=True):
    """Run a shell command and stream output."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not check, text=True)
    if check and result.returncode != 0:
        sys.exit(f"Command failed with code {result.returncode}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Download SOOP dataset from OpenNeuro")
    parser.add_argument(
        "--data-dir",
        default="/home/ubuntu/data/SOOP_Dataset",
        help="Root directory to store dataset (default: /home/ubuntu/data/SOOP_Dataset)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = os.path.join(data_dir, "raw")
    derivatives_dir = os.path.join(data_dir, "derivatives")
    masks_dir = os.path.join(data_dir, "masks")

    # --- Create directories ---
    for d in [raw_dir, derivatives_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"Directories created under {data_dir}")

    # --- Install awscli if missing ---
    if subprocess.run(["which", "aws"], capture_output=True).returncode != 0:
        print("Installing awscli...")
        subprocess.run([sys.executable, "-m", "pip", "install", "awscli", "-q"], check=True)
    print("AWS CLI ready")

    # --- Download participants.tsv ---
    print("Downloading participants.tsv...")
    run([
        "aws", "s3", "cp",
        "s3://openneuro.org/ds004889/participants.tsv",
        data_dir,
        "--no-sign-request",
    ])

    # --- List subjects ---
    result = subprocess.run(
        ["aws", "s3", "ls", "s3://openneuro.org/ds004889/", "--no-sign-request"],
        capture_output=True, text=True,
    )
    subjects = [
        line.split()[-1].strip("/")
        for line in result.stdout.splitlines()
        if "sub-" in line
    ]
    print(f"Found {len(subjects)} subjects")

    # --- Download each subject ---
    for i, subject in enumerate(subjects, 1):
        print(f"\nDownloading {subject} ({i}/{len(subjects)})...")
        run([
            "aws", "s3", "sync",
            f"s3://openneuro.org/ds004889/{subject}",
            os.path.join(raw_dir, subject),
            "--no-sign-request",
        ])
        print(f"  {subject} done")

    # --- Download derivatives (lesion masks) ---
    print("\nDownloading lesion masks (derivatives)...")
    run([
        "aws", "s3", "sync",
        "s3://openneuro.org/ds004889/derivatives",
        derivatives_dir,
        "--no-sign-request",
    ])

    print(f"\nAll done! SOOP dataset saved to {data_dir}")


if __name__ == "__main__":
    main()
