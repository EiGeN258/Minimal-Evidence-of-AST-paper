#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch wrapper for the ORIGINAL pseudo_gps_validate.py, preserving the user's
original raw/GT inputs whenever possible.

Design goal
-----------
For strict comparability against the user's AirIMU / Ours experiments:
- Prefer EXACTLY the same raw file used before:
    <orig_root>/<seq>/mav0/imu0/data.csv
- Prefer EXACTLY the same GT file used before:
    <orig_root>/<seq>/mav0/state_groundtruth_estimate0/data.csv
- Only if those files are missing, fall back to converting:
    imu.txt / gt.txt
- DENOISE always comes from DIDO merged IMU csv:
    <net_imu_dir>/<seq>_imu_net.csv

Why this matters
----------------
The original pseudo_gps_validate.py consumes raw/gt mechanically by column order.
If we re-export or reorder GT, the raw baseline may shift. Therefore, when the
original data.csv files exist, we pass them through directly, unchanged.
"""

from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_list_file(path: Path) -> List[str]:
    seqs = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            seqs.append(s)
    return seqs


def collect_sequences(args) -> List[str]:
    if args.seq:
        return [args.seq]
    if args.seqs:
        return [s.strip() for s in args.seqs.split(",") if s.strip()]
    if args.list_file:
        return read_list_file(Path(args.list_file))
    raise ValueError("Provide one of --seq / --seqs / --list_file")


def _iter_numeric_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            yield vals


def count_numeric_rows(path: Path) -> int:
    return sum(1 for _ in _iter_numeric_rows(path))


def export_raw_clean(raw_txt_path: Path, out_path: Path) -> int:
    """
    raw txt -> cleaned raw csv expected by original script.
    Input:
      8 cols: idx, timestamp_sec, wx, wy, wz, ax, ay, az
      7 cols: timestamp_sec, wx, wy, wz, ax, ay, az
    Output:
      timestamp_ns, wx, wy, wz, ax, ay, az
    """
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("# timestamp_ns, ang_vel_x, ang_vel_y, ang_vel_z, lin_acc_x, lin_acc_y, lin_acc_z\n")
        for vals in _iter_numeric_rows(raw_txt_path):
            if len(vals) == 8:
                ts_sec = vals[1]
                wx, wy, wz, ax, ay, az = vals[2:8]
            elif len(vals) == 7:
                ts_sec = vals[0]
                wx, wy, wz, ax, ay, az = vals[1:7]
            else:
                continue
            ts_ns = int(round(ts_sec * 1e9))
            f.write(f"{ts_ns},{wx:.9f},{wy:.9f},{wz:.9f},{ax:.9f},{ay:.9f},{az:.9f}\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No valid raw rows parsed from {raw_txt_path}")
    return n


def export_gt_clean(gt_txt_path: Path, out_path: Path) -> int:
    """
    gt txt -> cleaned gt csv expected by original script.
    Input:
      9 cols: idx, timestamp_sec, tx, ty, tz, qx, qy, qz, qw
      8 cols: timestamp_sec, tx, ty, tz, qx, qy, qz, qw
    Output:
      timestamp_ns, tx, ty, tz, qx, qy, qz, qw
    """
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("# timestamp_ns, tx, ty, tz, qx, qy, qz, qw\n")
        for vals in _iter_numeric_rows(gt_txt_path):
            if len(vals) == 9:
                ts_sec = vals[1]
                tx, ty, tz, qx, qy, qz, qw = vals[2:9]
            elif len(vals) == 8:
                ts_sec = vals[0]
                tx, ty, tz, qx, qy, qz, qw = vals[1:8]
            else:
                continue
            ts_ns = int(round(ts_sec * 1e9))
            f.write(f"{ts_ns},{tx:.9f},{ty:.9f},{tz:.9f},{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f}\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No valid GT rows parsed from {gt_txt_path}")
    return n


def export_denoise_csv(net_csv_path: Path, out_path: Path) -> int:
    """
    DIDO merged IMU:
      timestamp, wx, wy, wz, ax, ay, az   (absolute sec)
    ->
    original pseudo_gps_validate.py denoise format:
      timestamp_s(relative), wx, wy, wz, ax, ay, az
    """
    arr = np.loadtxt(net_csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise RuntimeError(f"Expected >=7 columns in {net_csv_path}, got {arr.shape[1]}")
    t_abs = arr[:, 0]
    gyro = arr[:, 1:4]
    acc = arr[:, 4:7]
    t_rel = t_abs - t_abs[0]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("# timestamp_s, ang_vel_x, ang_vel_y, ang_vel_z, lin_acc_x, lin_acc_y, lin_acc_z\n")
        for t, g, a in zip(t_rel, gyro, acc):
            f.write(f"{t:.9f},{g[0]:.9f},{g[1]:.9f},{g[2]:.9f},{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}\n")
    return int(len(arr))


def flatten_summary(d: dict, path: str, default=""):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def resolve_inputs(args, seq: str, tmp_dir: Path) -> Tuple[Path, Path, Path, dict]:
    seq_root = Path(args.orig_root) / seq / "mav0"

    raw_data_csv = seq_root / "imu0" / "data.csv"
    gt_data_csv = seq_root / "state_groundtruth_estimate0" / "data.csv"
    raw_txt = seq_root / "imu0" / "imu.txt"
    gt_txt = seq_root / "state_groundtruth_estimate0" / "gt.txt"
    net_csv = Path(args.net_imu_dir) / f"{seq}_imu_net.csv"

    if not net_csv.exists():
        raise FileNotFoundError(f"Missing DIDO merged IMU file: {net_csv}")

    meta = {
        "raw_source_mode": "",
        "gt_source_mode": "",
        "raw_source_path": "",
        "gt_source_path": "",
        "net_csv": str(net_csv),
    }

    # raw: prefer exact original data.csv
    if raw_data_csv.exists():
        raw_input = raw_data_csv
        n_raw = count_numeric_rows(raw_input)
        meta["raw_source_mode"] = "direct_data_csv"
        meta["raw_source_path"] = str(raw_input)
    elif raw_txt.exists():
        raw_clean = tmp_dir / "raw_clean.csv"
        n_raw = export_raw_clean(raw_txt, raw_clean)
        raw_input = raw_clean
        meta["raw_source_mode"] = "fallback_from_imu_txt"
        meta["raw_source_path"] = str(raw_txt)
        meta["raw_clean"] = str(raw_clean)
    else:
        raise FileNotFoundError(
            f"Missing raw IMU source. Tried: {raw_data_csv} and {raw_txt}"
        )

    # gt: prefer exact original data.csv
    if gt_data_csv.exists():
        gt_input = gt_data_csv
        n_gt = count_numeric_rows(gt_input)
        meta["gt_source_mode"] = "direct_data_csv"
        meta["gt_source_path"] = str(gt_input)
    elif gt_txt.exists():
        gt_clean = tmp_dir / "gt_clean.csv"
        n_gt = export_gt_clean(gt_txt, gt_clean)
        gt_input = gt_clean
        meta["gt_source_mode"] = "fallback_from_gt_txt"
        meta["gt_source_path"] = str(gt_txt)
        meta["gt_clean"] = str(gt_clean)
    else:
        raise FileNotFoundError(
            f"Missing GT source. Tried: {gt_data_csv} and {gt_txt}"
        )

    denoise_csv = tmp_dir / "denoise.csv"
    n_den = export_denoise_csv(net_csv, denoise_csv)
    meta["denoise_csv"] = str(denoise_csv)

    counts = {
        "raw_rows": int(n_raw),
        "gt_rows": int(n_gt),
        "denoise_rows": int(n_den),
    }
    return raw_input, gt_input, denoise_csv, {**meta, **counts}


def run_one(args, seq: str):
    seq_out = Path(args.outdir) / seq
    seq_out.mkdir(parents=True, exist_ok=True)

    tmp_dir = seq_out / "_strict_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_input, gt_input, denoise_csv, input_meta = resolve_inputs(args, seq, tmp_dir)

    cmd = [
        sys.executable, str(Path(args.original_script)),
        "--raw", str(raw_input),
        "--denoise", str(denoise_csv),
        "--gt", str(gt_input),
        "--outdir", str(seq_out),
        "--raw_label", args.raw_label,
        "--denoise_label", args.denoise_label,
        "--gps_period", str(args.gps_period),
        "--sigma_p", str(args.sigma_p),
        "--mode", args.mode,
        "--denoise_time_offset", str(args.denoise_time_offset),
        "--max_shift_sec", str(args.max_shift_sec),
        "--gravity", str(args.gravity),
        "--sigma_g", str(args.sigma_g),
        "--sigma_a", str(args.sigma_a),
        "--sigma_bg", str(args.sigma_bg),
        "--sigma_ba", str(args.sigma_ba),
        "--seed", str(args.seed),
    ]
    if args.add_gps_noise:
        cmd.append("--add_gps_noise")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    result = {
        "seq": seq,
        "returncode": int(proc.returncode),
        "input_counts": {
            "raw_rows": int(input_meta["raw_rows"]),
            "gt_rows": int(input_meta["gt_rows"]),
            "denoise_rows": int(input_meta["denoise_rows"]),
        },
        "input_paths": {
            "raw": str(raw_input),
            "gt": str(gt_input),
            "denoise_csv": str(denoise_csv),
            "raw_source_mode": input_meta["raw_source_mode"],
            "gt_source_mode": input_meta["gt_source_mode"],
            "raw_source_path": input_meta["raw_source_path"],
            "gt_source_path": input_meta["gt_source_path"],
            "net_csv": input_meta["net_csv"],
        },
    }
    if "raw_clean" in input_meta:
        result["input_paths"]["raw_clean"] = input_meta["raw_clean"]
    if "gt_clean" in input_meta:
        result["input_paths"]["gt_clean"] = input_meta["gt_clean"]

    summary_path = seq_out / "summary.json"
    if summary_path.exists():
        try:
            result.update(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception as e:
            result["summary_parse_error"] = f"{type(e).__name__}: {e}"

    if proc.returncode != 0:
        result["status"] = "failed"
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        raise RuntimeError(
            f"Original script failed for {seq}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    result["status"] = "ok"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_script", required=True,
                    help="Path to the user's ORIGINAL pseudo_gps_validate.py")
    ap.add_argument("--orig_root", required=True,
                    help="Root containing original per-sequence folders under <orig_root>/<seq>/mav0")
    ap.add_argument("--net_imu_dir", required=True,
                    help="DIDO merged IMU directory, e.g. .../net_imu_merged")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--seq", default="")
    ap.add_argument("--seqs", default="", help="Comma-separated sequence names")
    ap.add_argument("--list_file", default="", help="Text file with one sequence per line, e.g. dataset/test.txt")

    ap.add_argument("--raw_label", type=str, default="Raw")
    ap.add_argument("--denoise_label", type=str, default="DIDO")
    ap.add_argument("--gps_period", type=float, default=0.5)
    ap.add_argument("--sigma_p", type=float, default=0.1)
    ap.add_argument("--mode", choices=["drift", "eskf", "both"], default="both")
    ap.add_argument("--denoise_time_offset", type=float, default=0.0)
    ap.add_argument("--max_shift_sec", type=float, default=0.2)
    ap.add_argument("--gravity", type=float, default=-9.81)
    ap.add_argument("--sigma_g", type=float, default=0.02)
    ap.add_argument("--sigma_a", type=float, default=0.2)
    ap.add_argument("--sigma_bg", type=float, default=0.002)
    ap.add_argument("--sigma_ba", type=float, default=0.02)
    ap.add_argument("--add_gps_noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seqs = collect_sequences(args)
    all_summary = []

    for seq in seqs:
        try:
            res = run_one(args, seq)
            all_summary.append(res)
            print(f"[OK] {seq}")
        except Exception as e:
            fail = {"seq": seq, "status": "failed", "error": f"{type(e).__name__}: {e}"}
            all_summary.append(fail)
            print(f"[FAIL] {seq}: {fail['error']}")

    (outdir / "summary_all.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    keys = [
        "seq", "status",
        "input_counts.raw_rows", "input_counts.gt_rows", "input_counts.denoise_rows",
        "gt_span_s", "raw_cropped_N", "dt_med_ms",
        "lag_refine_shift_samples", "lag_refine_shift_ms", "lag_refine_corr",
        "drift_raw.mean", "drift_denoised.mean",
        "eskf_poserr_raw.mean", "eskf_poserr_denoised.mean",
        "eskf_innov_raw.mean", "eskf_innov_denoised.mean",
        "eskf_nis_raw.mean", "eskf_nis_denoised.mean"
    ]
    with (outdir / "summary_all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for d in all_summary:
            row = []
            for k in keys:
                if k == "status":
                    row.append(d.get("status", ""))
                else:
                    row.append(flatten_summary(d, k, ""))
            w.writerow(row)

    print("Done.")
    print("Outputs:")
    print("  ", outdir / "summary_all.json")
    print("  ", outdir / "summary_all.csv")


if __name__ == "__main__":
    main()
