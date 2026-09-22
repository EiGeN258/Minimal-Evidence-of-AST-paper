#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate DIDO-corrected IMU (net_imu_merged) on UZH-FPV sequences with pure IMU integration.

Style: mirrors your AirIMU visualize_state.py (Times New Roman + high-contrast palette). See your
uploaded visualize_state.py / evaluate_state.py for the original style conventions.

Inputs
------
Dataset root (DIDO format):
  <dataset_root>/<seq>/data.hdf5 with datasets: ts, acc, gyr, gt_p, gt_v, gt_q (wxyz)

Corrected IMU CSVs (merged):
  <net_imu_dir>/<seq>_imu_net.csv with header:
    timestamp,wx,wy,wz,ax,ay,az

Outputs
-------
out_dir/
  summary_metrics.csv
  summary_metrics.json
  <seq>/
    traj_xy.png
    pos_vel_error_time.png
    rel_pos_vel_error_window.png
    rot_error_time.png
    meta.txt

Metric definitions (aligned with AirIMU evaluate_state.py convention)
---------------------------------------------------------------------
ATE / AVE: mean position/velocity norm error over the full (non-reset) integration.
RTE / RVE: by default, mean relative position/velocity errors over a fixed horizon (seqlen samples) computed from the full integrated trajectory. Use --rte_mode reset for reset-window endpoint errors.

By default we integrate orientation from gyro and use it for gravity compensation.
For debugging you can enable --use_gt_rot (use gt_q for gravity compensation).

Example (PowerShell)
--------------------
python eval_dido_uzh_net_imu.py `
  --dataset_root "S:/DIDO-main/dataset" `
  --net_imu_dir  "S:/DIDO-main/Rotation_ekf/output/net_imu_merged" `
  --out_dir      "S:/DIDO-main/Rotation_ekf/output/eval_net_imu" `
  --seqs "UZH_Indoorforward_9_davis,UZH_Outdoorforward_3_davis,UZH_Indoor45_9_davis,UZH_Outdoor45_1_davis" `
  --seqlen 200 `
  --gravity "0,0,-9.81"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------
# Plot style (mirrors your visualize_state.py)
# -----------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "axes.labelpad": 2.0,
    "axes.titlepad": 6.0,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
})

COL_WINE  = "#DF075E"
COL_ORANGE = "#F46F44"
COL_BLUE   = "#7489C3"
COL_RAW    = COL_BLUE
COL_GT     = COL_ORANGE
COL_NET    = COL_WINE

LW_RAW = 1.5
LW_GT  = 1.5
LW_NET = 1.8
GRID_ALPHA = 0.35
GRID_LW = 0.8


# -----------------------------
# Quaternion ops (wxyz)
# -----------------------------
def q_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return q / n


def q_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)


def q_to_R(q: np.ndarray) -> np.ndarray:
    q = q_norm(q)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=np.float64)
    return R


def omega_to_dq(omega: np.ndarray, dt: float) -> np.ndarray:
    wx, wy, wz = omega.astype(np.float64)
    theta = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.array([wx, wy, wz], dtype=np.float64)
    axis /= (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * theta
    s = math.sin(half)
    return q_norm(np.array([math.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64))


def q_angle_err(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    q_pred = q_norm(q_pred)
    q_gt = q_norm(q_gt)
    q_e = q_mul(q_conj(q_pred), q_gt)
    w = float(np.clip(abs(q_e[0]), -1.0, 1.0))
    return 2.0 * math.acos(w)


# -----------------------------
# IO
# -----------------------------
def load_hdf5_seq(h5_path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        def get(name):
            if name not in f:
                raise KeyError(f"Missing dataset '{name}' in {h5_path}")
            return np.array(f[name])
        data = {
            "ts": get("ts").astype(np.float64).reshape(-1),
            "acc": get("acc").astype(np.float64),
            "gyr": get("gyr").astype(np.float64),
            "gt_p": get("gt_p").astype(np.float64),
            "gt_v": get("gt_v").astype(np.float64),
            "gt_q": get("gt_q").astype(np.float64),
        }
    data["gt_q"] = q_norm(data["gt_q"])
    return data


def load_net_imu_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise ValueError(f"Expected >=7 columns in {path}, got {arr.shape[1]}")
    ts = arr[:, 0].reshape(-1)
    gyr = arr[:, 1:4]
    acc = arr[:, 4:7]
    return ts, gyr, acc


# -----------------------------
# Integration and evaluation
# -----------------------------
def integrate_full(ts: np.ndarray,
                   acc: np.ndarray,
                   gyr: np.ndarray,
                   init_p: np.ndarray,
                   init_v: np.ndarray,
                   init_q: np.ndarray,
                   gravity: np.ndarray,
                   use_gt_rot: bool = False,
                   gt_q: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(ts)
    p = np.zeros((N, 3), dtype=np.float64)
    v = np.zeros((N, 3), dtype=np.float64)
    q = np.zeros((N, 4), dtype=np.float64)
    p[0] = init_p
    v[0] = init_v
    q[0] = q_norm(init_q)

    for k in range(N - 1):
        dt = float(ts[k + 1] - ts[k])
        if dt <= 0:
            dt = 1e-3
        q_rot = gt_q[k] if (use_gt_rot and gt_q is not None) else q[k]
        R = q_to_R(q_rot)
        a_world = R @ acc[k] + gravity
        v[k + 1] = v[k] + a_world * dt
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a_world * dt * dt
        dq = omega_to_dq(gyr[k], dt)
        q[k + 1] = q_norm(q_mul(q[k], dq))
    return p, v, q


def integrate_reset_windows(ts: np.ndarray,
                            acc: np.ndarray,
                            gyr: np.ndarray,
                            gt_p: np.ndarray,
                            gt_v: np.ndarray,
                            gt_q: np.ndarray,
                            gravity: np.ndarray,
                            seqlen: int,
                            use_gt_rot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    N = len(ts)
    pos_err = []
    vel_err = []
    for s in range(0, N - seqlen, seqlen):
        e = s + seqlen - 1
        p_pred, v_pred, _ = integrate_full(
            ts[s:e+1], acc[s:e+1], gyr[s:e+1],
            init_p=gt_p[s], init_v=gt_v[s], init_q=gt_q[s],
            gravity=gravity,
            use_gt_rot=use_gt_rot,
            gt_q=gt_q[s:e+1] if use_gt_rot else None,
        )
        pos_err.append(float(np.linalg.norm(p_pred[-1] - gt_p[e])))
        vel_err.append(float(np.linalg.norm(v_pred[-1] - gt_v[e])))
    return np.asarray(pos_err), np.asarray(vel_err)



def compute_relative_errors_from_full(p_pred: np.ndarray, v_pred: np.ndarray,
                                      gt_p: np.ndarray, gt_v: np.ndarray,
                                      seqlen: int, step: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relative position/velocity errors from the full integrated trajectory (no reset integration).

    For each start i and end j=i+seqlen:
        RTE_i = || (p_pred[j]-p_pred[i]) - (gt_p[j]-gt_p[i]) ||
        RVE_i = || (v_pred[j]-v_pred[i]) - (gt_v[j]-gt_v[i]) ||
    """
    N = len(p_pred)
    if step is None or step <= 0:
        step = seqlen
    rte, rve = [], []
    for i in range(0, N - seqlen, step):
        j = i + seqlen
        dp_err = (p_pred[j] - p_pred[i]) - (gt_p[j] - gt_p[i])
        dv_err = (v_pred[j] - v_pred[i]) - (gt_v[j] - gt_v[i])
        rte.append(float(np.linalg.norm(dp_err)))
        rve.append(float(np.linalg.norm(dv_err)))
    return np.asarray(rte), np.asarray(rve)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plotting
# -----------------------------
def plot_traj_xy(seq: str, outdir: Path,
                 gt_p: np.ndarray, p_raw: np.ndarray, p_net: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)
    ax.plot(gt_p[:, 0], gt_p[:, 1], color=COL_GT, lw=LW_GT, label="Ground Truth")
    ax.plot(p_raw[:, 0], p_raw[:, 1], color=COL_RAW, lw=LW_RAW, label="Raw")
    ax.plot(p_net[:, 0], p_net[:, 1], color=COL_NET, lw=LW_NET, label="Net-IMU")
    ax.set_title(f"{seq} | Trajectory (XY)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outdir / "traj_xy.png", bbox_inches="tight")
    plt.close(fig)


def plot_pos_vel_error(seq: str, outdir: Path,
                       ts: np.ndarray,
                       pos_err_raw: np.ndarray, pos_err_net: np.ndarray,
                       vel_err_raw: np.ndarray, vel_err_net: np.ndarray) -> None:
    t = ts - ts[0]
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.4, 6.2), dpi=300)
    fig.suptitle(f"{seq} | Position / Velocity error vs time", y=0.98)

    axs[0].plot(t, pos_err_raw, color=COL_RAW, lw=LW_RAW, label="Raw")
    axs[0].plot(t, pos_err_net, color=COL_NET, lw=LW_NET, label="Net-IMU")
    axs[0].set_ylabel("Position error (m)")
    axs[0].grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    axs[0].legend(loc="upper left", framealpha=0.9)

    axs[1].plot(t, vel_err_raw, color=COL_RAW, lw=LW_RAW, label="Raw")
    axs[1].plot(t, vel_err_net, color=COL_NET, lw=LW_NET, label="Net-IMU")
    axs[1].set_ylabel("Velocity error (m/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    axs[1].legend(loc="upper left", framealpha=0.9)

    for ax in axs:
        ax.yaxis.set_label_coords(-0.095, 0.5)
    fig.tight_layout(rect=[0.06, 0.04, 0.995, 0.95])
    fig.savefig(outdir / "pos_vel_error_time.png", bbox_inches="tight")
    plt.close(fig)


def plot_rot_error(seq: str, outdir: Path, ts: np.ndarray,
                   rot_err_raw: np.ndarray, rot_err_net: np.ndarray) -> None:
    t = ts - ts[0]
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=300)
    ax.plot(t, rot_err_raw * 180.0 / np.pi, color=COL_RAW, lw=LW_RAW, label="Raw")
    ax.plot(t, rot_err_net * 180.0 / np.pi, color=COL_NET, lw=LW_NET, label="Net-IMU")
    ax.set_title(f"{seq} | Rotation error vs time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rotation error (deg)")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / "rot_error_time.png", bbox_inches="tight")
    plt.close(fig)


def plot_relative_errors(seq: str, outdir: Path,
                         rte_raw: np.ndarray, rte_net: np.ndarray,
                         rve_raw: np.ndarray, rve_net: np.ndarray,
                         seqlen: int, dt_med: float) -> None:
    x = np.arange(len(rte_raw))
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.4, 6.2), dpi=300)
    fig.suptitle(f"{seq} | Relative errors (window={seqlen} samples ≈ {seqlen*dt_med:.3f}s)", y=0.98)

    axs[0].plot(x, rte_raw, color=COL_RAW, lw=LW_RAW, label="Raw")
    axs[0].plot(x, rte_net, color=COL_NET, lw=LW_NET, label="Net-IMU")
    axs[0].set_ylabel("RTE (m)")
    axs[0].grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    axs[0].legend(loc="upper left", framealpha=0.9)

    axs[1].plot(x, rve_raw, color=COL_RAW, lw=LW_RAW, label="Raw")
    axs[1].plot(x, rve_net, color=COL_NET, lw=LW_NET, label="Net-IMU")
    axs[1].set_ylabel("RVE (m/s)")
    axs[1].set_xlabel("Window index")
    axs[1].grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    axs[1].legend(loc="upper left", framealpha=0.9)

    for ax in axs:
        ax.yaxis.set_label_coords(-0.095, 0.5)
    fig.tight_layout(rect=[0.06, 0.04, 0.995, 0.95])
    fig.savefig(outdir / "rel_pos_vel_error_window.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("gravity must be like '0,0,-9.81'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--net_imu_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seqs", type=str, default="")
    ap.add_argument("--list_file", type=str, default="")
    ap.add_argument("--seqlen", type=int, default=200)
    ap.add_argument("--gravity", type=str, default="0,0,-9.81")
    ap.add_argument("--use_gt_rot", action="store_true")
    ap.add_argument("--rte_mode", type=str, default="relative", choices=["relative", "reset"],
                    help="RTE/RVE mode: relative (full-trajectory horizon errors) or reset (reset-window endpoint errors).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    net_imu_dir = Path(args.net_imu_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    gravity = parse_vec3(args.gravity)

    if args.seqs.strip():
        seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    elif args.list_file.strip():
        seqs = [ln.strip() for ln in Path(args.list_file).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    else:
        raise ValueError("Provide either --seqs or --list_file")

    all_rows = []

    for seq in seqs:
        h5_path = dataset_root / seq / "data.hdf5"
        net_csv = net_imu_dir / f"{seq}_imu_net.csv"
        if not h5_path.exists():
            print(f"[SKIP] Missing {h5_path}")
            continue
        if not net_csv.exists():
            print(f"[SKIP] Missing {net_csv}")
            continue

        d = load_hdf5_seq(h5_path)
        ts = d["ts"]
        acc_raw = d["acc"]
        gyr_raw = d["gyr"]
        gt_p = d["gt_p"]
        gt_v = d["gt_v"]
        gt_q = d["gt_q"]

        ts2, gyr_net, acc_net = load_net_imu_csv(net_csv)

        n = min(len(ts), len(ts2), len(acc_raw), len(gyr_raw), len(gt_p), len(gt_v), len(gt_q), len(acc_net), len(gyr_net))
        ts = ts[:n]
        acc_raw = acc_raw[:n]
        gyr_raw = gyr_raw[:n]
        gt_p = gt_p[:n]
        gt_v = gt_v[:n]
        gt_q = gt_q[:n]
        acc_net = acc_net[:n]
        gyr_net = gyr_net[:n]

        dt_med = float(np.median(np.diff(ts))) if n > 2 else float("nan")

        init_p, init_v, init_q = gt_p[0], gt_v[0], gt_q[0]
        p_raw, v_raw, q_raw = integrate_full(ts, acc_raw, gyr_raw, init_p, init_v, init_q, gravity,
                                             use_gt_rot=args.use_gt_rot, gt_q=gt_q if args.use_gt_rot else None)
        p_net, v_net, q_net = integrate_full(ts, acc_net, gyr_net, init_p, init_v, init_q, gravity,
                                             use_gt_rot=args.use_gt_rot, gt_q=gt_q if args.use_gt_rot else None)

        pos_err_raw = np.linalg.norm(p_raw - gt_p, axis=1)
        pos_err_net = np.linalg.norm(p_net - gt_p, axis=1)
        vel_err_raw = np.linalg.norm(v_raw - gt_v, axis=1)
        vel_err_net = np.linalg.norm(v_net - gt_v, axis=1)

        rot_err_raw = np.array([q_angle_err(q_raw[i], gt_q[i]) for i in range(n)], dtype=np.float64)
        rot_err_net = np.array([q_angle_err(q_net[i], gt_q[i]) for i in range(n)], dtype=np.float64)

        ate_raw = float(pos_err_raw.mean())
        ave_raw = float(vel_err_raw.mean())
        ate_net = float(pos_err_net.mean())
        ave_net = float(vel_err_net.mean())

        if args.rte_mode == "reset":
            rte_raw, rve_raw = integrate_reset_windows(ts, acc_raw, gyr_raw, gt_p, gt_v, gt_q, gravity, args.seqlen, args.use_gt_rot)
            rte_net, rve_net = integrate_reset_windows(ts, acc_net, gyr_net, gt_p, gt_v, gt_q, gravity, args.seqlen, args.use_gt_rot)
        else:
            rte_raw, rve_raw = compute_relative_errors_from_full(p_raw, v_raw, gt_p, gt_v, args.seqlen, step=args.seqlen)
            rte_net, rve_net = compute_relative_errors_from_full(p_net, v_net, gt_p, gt_v, args.seqlen, step=args.seqlen)

        rte_raw_m = float(rte_raw.mean()) if len(rte_raw) else float("nan")
        rve_raw_m = float(rve_raw.mean()) if len(rve_raw) else float("nan")
        rte_net_m = float(rte_net.mean()) if len(rte_net) else float("nan")
        rve_net_m = float(rve_net.mean()) if len(rve_net) else float("nan")

        seq_out = out_dir / seq
        ensure_dir(seq_out)
        plot_traj_xy(seq, seq_out, gt_p, p_raw, p_net)
        plot_pos_vel_error(seq, seq_out, ts, pos_err_raw, pos_err_net, vel_err_raw, vel_err_net)
        plot_rot_error(seq, seq_out, ts, rot_err_raw, rot_err_net)
        plot_relative_errors(seq, seq_out, rte_raw, rte_net, rve_raw, rve_net, args.seqlen, dt_med)

        meta = {
            "seq": seq,
            "N": int(n),
            "dt_median": dt_med,
            "gravity": gravity.tolist(),
            "use_gt_rot": bool(args.use_gt_rot),
            "rte_mode": args.rte_mode,
            "paths": {"h5": str(h5_path), "net_csv": str(net_csv)},
        }
        (seq_out / "meta.txt").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        row = {
            "seq": seq,
            "N": int(n),
            "dt_med(s)": dt_med,
            "ATE(raw)": ate_raw,
            "AVE(raw)": ave_raw,
            "RTE(raw)": rte_raw_m,
            "RVE(raw)": rve_raw_m,
            "ATE(net)": ate_net,
            "AVE(net)": ave_net,
            "RTE(net)": rte_net_m,
            "RVE(net)": rve_net_m,
        }
        all_rows.append(row)

        print(f"[OK] {seq} | ATE(raw/net)={ate_raw:.3f}/{ate_net:.3f} m | "
              f"AVE(raw/net)={ave_raw:.3f}/{ave_net:.3f} m/s | "
              f"RTE(raw/net)={rte_raw_m:.3f}/{rte_net_m:.3f} m | "
              f"RVE(raw/net)={rve_raw_m:.3f}/{rve_net_m:.3f} m/s")

    if all_rows:
        keys = list(all_rows[0].keys())
        lines = [",".join(keys)]
        for r in all_rows:
            lines.append(",".join([str(r[k]) for k in keys]))
        (out_dir / "summary_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "summary_metrics.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
