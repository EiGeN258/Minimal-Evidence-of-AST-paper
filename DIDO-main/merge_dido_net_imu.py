#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge DIDO Rotation_ekf net_acc/net_gyr outputs with timestamps into a single per-sequence IMU file.

Inputs (your case)
------------------
Dataset root:
  S:/DIDO-main/dataset/<seq>/data.hdf5  (contains dataset 'ts')
Net outputs:
  S:/DIDO-main/Rotation_ekf/output/net_acc/<seq>_acc.txt
  S:/DIDO-main/Rotation_ekf/output/net_gyr/<seq>_gyr.txt

Outputs
-------
out_dir/
  <seq>_imu_net.csv     # timestamp, wx, wy, wz, ax, ay, az (comma-separated)
  merge_report.txt      # quick report about lengths, fallbacks, etc.

Notes
-----
- Robust to comma-separated or whitespace-separated txt.
- If net files contain more than 3 columns (rare), it takes the last 3 columns by default.
- If lengths mismatch, it crops all to the minimum length and reports it.

Usage example (PowerShell)
--------------------------
python merge_dido_net_imu.py `
  --dataset_root "S:/DIDO-main/dataset" `
  --net_acc_dir  "S:/DIDO-main/Rotation_ekf/output/net_acc" `
  --net_gyr_dir  "S:/DIDO-main/Rotation_ekf/output/net_gyr" `
  --list_file    "S:/DIDO-main/dataset/gen_list.txt" `
  --out_dir      "S:/DIDO-main/Rotation_ekf/output/net_imu_merged"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import h5py


def read_list_file(path: Path) -> List[str]:
    names = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


def _load_txt_matrix(path: Path) -> np.ndarray:
    """Try comma-separated first, then whitespace. Return float64 ndarray."""
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except Exception:
        arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _take_last3(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] == 3:
        return arr
    if arr.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns, got {arr.shape[1]}")
    return arr[:, -3:]


def read_ts_from_hdf5(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "ts" not in f:
            raise KeyError(f"Dataset 'ts' not found in {h5_path}")
        ts = np.array(f["ts"], dtype=np.float64)
    return ts.reshape(-1)


def merge_one(seq: str, dataset_root: Path, net_acc_dir: Path, net_gyr_dir: Path) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    h5_path = dataset_root / seq / "data.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing: {h5_path}")

    ts = read_ts_from_hdf5(h5_path)

    acc_path = net_acc_dir / f"{seq}_acc.txt"
    gyr_path = net_gyr_dir / f"{seq}_gyr.txt"
    if not acc_path.exists():
        raise FileNotFoundError(f"Missing: {acc_path}")
    if not gyr_path.exists():
        raise FileNotFoundError(f"Missing: {gyr_path}")

    acc_raw = _load_txt_matrix(acc_path)
    gyr_raw = _load_txt_matrix(gyr_path)
    acc = _take_last3(acc_raw)
    gyr = _take_last3(gyr_raw)

    n = min(len(ts), len(acc), len(gyr))
    ts = ts[:n]
    acc = acc[:n]
    gyr = gyr[:n]

    merged = np.column_stack([ts, gyr, acc])  # [t, wx, wy, wz, ax, ay, az]
    return merged, (len(ts), len(acc), len(gyr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Dataset root containing <seq>/data.hdf5 (with dataset 'ts').")
    ap.add_argument("--net_acc_dir", type=str, required=True,
                    help="Directory containing <seq>_acc.txt")
    ap.add_argument("--net_gyr_dir", type=str, required=True,
                    help="Directory containing <seq>_gyr.txt")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for merged imu files.")
    ap.add_argument("--list_file", type=str, default="",
                    help="Optional list file (gen_list.txt / test.txt). If not set, auto-detect seq dirs.")
    ap.add_argument("--fmt", type=str, default="%.9f",
                    help="Numeric format for saving csv.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    net_acc_dir = Path(args.net_acc_dir)
    net_gyr_dir = Path(args.net_gyr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_file.strip():
        seqs = read_list_file(Path(args.list_file))
    else:
        seqs = sorted([p.name for p in dataset_root.iterdir()
                       if p.is_dir() and (p / "data.hdf5").exists()])

    report_lines = []
    ok = 0
    for seq in seqs:
        try:
            merged, lens = merge_one(seq, dataset_root, net_acc_dir, net_gyr_dir)
            out_path = out_dir / f"{seq}_imu_net.csv"
            header = "timestamp,wx,wy,wz,ax,ay,az"
            np.savetxt(out_path, merged, delimiter=",", header=header, comments="", fmt=args.fmt)
            ok += 1
            report_lines.append(f"[OK] {seq}: rows={merged.shape[0]} (ts/acc/gyr={lens}) -> {out_path.name}")
        except Exception as e:
            report_lines.append(f"[FAIL] {seq}: {type(e).__name__}: {e}")

    report_path = out_dir / "merge_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Done. OK={ok}/{len(seqs)}. Report: {report_path}")


if __name__ == "__main__":
    main()
