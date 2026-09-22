#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


SEQUENCES = [
    "UZH_Indoor45_9_davis",
    "UZH_Indoorforward_9_davis",
    "UZH_Outdoor45_1_davis",
    "UZH_Outdoorforward_3_davis",
]

METHODS = {
    "Baseline": "Baseline_net_output",
    "Ours": "Ours_net_output",
}


def parse_seeds(text: str):
    out = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def read_numeric(path: Path, n_cols: int):
    ts = []
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",") if "," in s else s.split()
            if len(parts) < n_cols:
                continue
            parts = parts[:n_cols]
            try:
                token = parts[0].strip()
                try:
                    t = int(token)
                except Exception:
                    t = int(Decimal(token))
                row = [float(x) for x in parts[1:]]
            except Exception:
                continue
            ts.append(t)
            data.append(row)
    if not ts:
        raise RuntimeError(f"No numeric rows found: {path}")
    return np.asarray(ts, dtype=np.int64), np.asarray(data, dtype=np.float64)



def read_relative_seconds(path: Path, n_cols: int = 7):
    ts = []
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",") if "," in s else s.split()
            if len(parts) < n_cols:
                continue
            parts = parts[:n_cols]
            try:
                t = float(parts[0])
                row = [float(x) for x in parts[1:]]
            except Exception:
                continue
            ts.append(t)
            data.append(row)
    if not ts:
        raise RuntimeError(f"No numeric rows found: {path}")
    return np.asarray(ts, dtype=np.float64), np.asarray(data, dtype=np.float64)


def interp_to_time(t_target, t_src, data_src):
    out = np.zeros((len(t_target), data_src.shape[1]), dtype=np.float64)
    for j in range(data_src.shape[1]):
        out[:, j] = np.interp(t_target, t_src, data_src[:, j])
    return out


def estimate_sample_shift(x, y, max_shift):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x)
    best_s, best_c = 0, -1e18
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            n_ov = n - s
            if n_ov < 200:
                continue
            c = float(np.dot(x[:n_ov], y[s:s+n_ov])) / n_ov
        else:
            s2 = -s
            n_ov = n - s2
            if n_ov < 200:
                continue
            c = float(np.dot(x[s2:s2+n_ov], y[:n_ov])) / n_ov
        if c > best_c:
            best_c, best_s = c, s
    return best_s, best_c


def apply_shift(arr, shift):
    n = arr.shape[0]
    out = np.empty_like(arr)
    if shift == 0:
        return arr.copy()
    if shift > 0:
        out[:n-shift] = arr[shift:]
        out[n-shift:] = arr[-1:]
    else:
        s = -shift
        out[s:] = arr[:n-s]
        out[:s] = arr[:1]
    return out


def find_corrected_rel(method_root: Path, seq: str):
    seq_dir = method_root / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    exact = seq_dir / f"{seq}_denoised_rel_s.csv"
    if exact.is_file():
        return exact
    hits = sorted(seq_dir.glob("*_denoised_rel_s.csv"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Cannot resolve corrected relative-time IMU in {seq_dir}")


def find_corrected_abs_unused(method_root: Path, seq: str):
    seq_dir = method_root / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    exact = seq_dir / f"{seq}_denoised_abs_ns.csv"
    if exact.is_file():
        return exact
    hits = sorted(seq_dir.glob("*_denoised_abs_ns.csv"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Cannot resolve corrected absolute-time IMU in {seq_dir}")


def quat_normalize(q):
    n = float(np.linalg.norm(q))
    if n <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return quat_normalize(np.array([0.5*w[0], 0.5*w[1], 0.5*w[2], 1.0], dtype=np.float64))
    axis = w / theta
    half = 0.5 * theta
    s = math.sin(half)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(half)], dtype=np.float64)


def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)],
    ], dtype=np.float64)


def slerp(q0, q1, u):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return quat_normalize((1.0-u)*q0 + u*q1)
    theta0 = math.acos(dot)
    s0 = math.sin(theta0)
    theta = theta0 * u
    return quat_normalize(math.sin(theta0-theta)/s0*q0 + math.sin(theta)/s0*q1)


def skew(a):
    ax, ay, az = a
    return np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)


class GT:
    def __init__(self, ts_ns, gt7):
        self.t0_ns = int(ts_ns[0])
        self.t1_ns = int(ts_ns[-1])
        self.t = (ts_ns.astype(np.float64) - float(self.t0_ns)) * 1e-9
        self.p = gt7[:, 0:3].astype(np.float64)
        self.q = np.asarray([quat_normalize(x) for x in gt7[:, 3:7]], dtype=np.float64)
        for i in range(1, len(self.t)):
            if self.t[i] <= self.t[i-1]:
                self.t[i] = self.t[i-1] + 1e-6
        self.v = np.gradient(self.p, self.t, axis=0)

    def interp_vec3(self, tq, arr):
        return np.array([np.interp(tq, self.t, arr[:, i]) for i in range(3)], dtype=np.float64)

    def p_at(self, tq):
        return self.interp_vec3(tq, self.p)

    def v_at(self, tq):
        return self.interp_vec3(tq, self.v)

    def q_at(self, tq):
        if tq <= self.t[0]:
            return self.q[0].copy()
        if tq >= self.t[-1]:
            return self.q[-1].copy()
        i = int(np.searchsorted(self.t, tq) - 1)
        i = max(0, min(i, len(self.t)-2))
        t0, t1 = self.t[i], self.t[i+1]
        u = 0.0 if t1 == t0 else (tq-t0)/(t1-t0)
        return slerp(self.q[i], self.q[i+1], float(u))


def strapdown_step(p, v, q, w, a, dt):
    q = quat_normalize(quat_mul(q, so3_exp(w*dt)))
    R = quat_to_rot(q)
    a_w = R @ a + np.array([0.0, 0.0, -9.81], dtype=np.float64)
    p = p + v*dt + 0.5*a_w*(dt*dt)
    v = v + a_w*dt
    return p, v, q


def norm_ppf(p):
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0*math.log(1.0-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def chi2_ppf(p, k):
    z = norm_ppf(p)
    a = 2.0/(9.0*k)
    return k*(1.0-a+z*math.sqrt(a))**3


def outage_policy(span_s):
    if span_s >= 70.0:
        return 3, 8.5, 9.5, 4.0
    if span_s >= 25.0:
        return 1, 8.5, 9.5, 3.0
    return 1, 6.0, 7.0, 2.0


def build_schedule(t_end, seed):
    n, dmin, dmax, margin = outage_policy(t_end)
    cand = np.arange(1.0, t_end + 1e-9, 1.0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    windows = []
    attempts = 0
    while len(windows) < n and attempts < 5000:
        attempts += 1
        dur = float(rng.uniform(dmin, dmax))
        min_start = 1.0
        max_stop = max(min_start, t_end - 1.0)
        latest_start = max(min_start, max_stop - dur)
        if latest_start <= min_start:
            break
        st = float(rng.uniform(min_start, latest_start))
        ed = min(st + dur, t_end - 1e-6)
        if ed <= st:
            continue
        ok = True
        for a, b in windows:
            if not (ed + margin <= a or st >= b + margin):
                ok = False
                break
        if ok:
            windows.append((st, ed))
    windows.sort(key=lambda x: x[0])
    valid_mask = np.ones_like(cand, dtype=bool)
    for i, t in enumerate(cand):
        for a, b in windows:
            if a <= t <= b:
                valid_mask[i] = False
                break
    return cand[valid_mask], windows


def outage_elapsed(t, valid):
    if len(valid) == 0 or t <= 1.0:
        return 0.0
    idx = np.searchsorted(valid, t, side="right") - 1
    last_valid = float(valid[idx]) if idx >= 0 else 0.0
    return float(max(0.0, t - (last_valid + 1.0)))


def run_eskf(t, imu6, gt, seed):
    valid_times, windows = build_schedule(float(t[-1]), seed)

    p = gt.p_at(0.0)
    v = gt.v_at(0.0)
    q = gt.q_at(0.0)
    bg = np.zeros(3)
    ba = np.zeros(3)

    att0 = math.radians(8.0)
    P = np.diag(
        [1.5**2]*3 +
        [0.8**2]*3 +
        [att0**2]*3 +
        [0.05**2]*3 +
        [0.5**2]*3
    ).astype(np.float64)

    Qc = np.diag(
        [0.02**2]*3 +
        [0.2**2]*3 +
        [0.002**2]*3 +
        [0.02**2]*3
    ).astype(np.float64)

    R_base = (0.5**2) * np.eye(3, dtype=np.float64)

    pos_err = np.zeros(len(t), dtype=np.float64)
    nis_times = []
    nis_values = []

    up_idx = 0
    r_adapt_scale = 1.0
    reacq_left = 0
    nis_reject_hi = float(chi2_ppf(0.999, 3))

    for k in range(len(t)-1):
        tk1 = float(t[k+1])
        dt = float(t[k+1]-t[k])
        if dt <= 0:
            continue

        wm = imu6[k, 0:3]
        am = imu6[k, 3:6]
        w = wm - bg
        a = am - ba

        p, v, q = strapdown_step(p, v, q, w, a, dt)

        oe = outage_elapsed(tk1, valid_times)
        q_scale = 1.0
        if oe > 0.0:
            q_scale = 1.0 + 3.0*(oe/5.0)
            q_scale = float(min(max(1.0, q_scale), 10.0))

        Rwb = quat_to_rot(q)

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -Rwb @ skew(a)
        F[3:6, 12:15] = -Rwb
        F[6:9, 6:9] = -skew(w)
        F[6:9, 9:12] = -np.eye(3)

        G = np.zeros((15, 12))
        G[6:9, 0:3] = -np.eye(3)
        G[3:6, 3:6] = -Rwb
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        Phi = np.eye(15) + F*dt
        Qd = (G @ (Qc*q_scale) @ G.T) * dt
        P = Phi @ P @ Phi.T + Qd

        pos_err[k+1] = float(np.linalg.norm(p - gt.p_at(tk1)))

        while up_idx < len(valid_times) and tk1 + 1e-12 >= float(valid_times[up_idx]):
            t_up = float(valid_times[up_idx])
            prev_valid = float(valid_times[up_idx-1]) if up_idx > 0 else 0.0
            gap = max(0.0, t_up - prev_valid - 1.0)
            if gap > 0.5:
                reacq_left = 3

            if reacq_left > 0:
                step_idx = 3 - reacq_left
                reacq_scale = max(1.0, 9.0*(0.5**step_idx))
                reacq_left -= 1
            else:
                reacq_scale = 1.0

            z = gt.p_at(t_up)
            y = (z - p).reshape(3, 1)

            H = np.zeros((3, 15))
            H[:, 0:3] = np.eye(3)

            R_scale_eff = float(np.clip(reacq_scale*r_adapt_scale, 0.5, 25.0))
            Rm = R_base * R_scale_eff
            S = H @ P @ H.T + Rm
            Sinv = np.linalg.inv(S)
            innov_norm = float(np.linalg.norm(y))
            nis = float((y.T @ Sinv @ y)[0, 0])
            nis_ratio = nis / 3.0

            if nis_ratio > 1.5:
                inflate = float(np.clip(nis_ratio/1.5, 1.0, 25.0))
                R_scale_eff = float(np.clip(R_scale_eff*inflate, 0.5, 25.0))
                Rm = R_base * R_scale_eff
                S = H @ P @ H.T + Rm
                Sinv = np.linalg.inv(S)
                nis = float((y.T @ Sinv @ y)[0, 0])
                nis_ratio = nis / 3.0

            nis_times.append(t_up)
            nis_values.append(nis)

            beta = 0.20 if nis_ratio >= 1.0 else 0.05
            target_scale = float(np.clip(nis_ratio, 0.5, 25.0))
            r_adapt_scale = float(np.clip(
                (1.0-beta)*r_adapt_scale + beta*target_scale,
                0.5,
                25.0
            ))

            innov_sigma = float(math.sqrt(max(1e-12, np.trace(S))))
            innov_reject = innov_norm > 6.0*innov_sigma
            nis_reject = nis > nis_reject_hi

            if innov_reject or nis_reject:
                up_idx += 1
                continue

            K = P @ H.T @ Sinv
            dx = (K @ y).reshape(-1)

            p = p + dx[0:3]
            v = v + dx[3:6]
            q = quat_normalize(quat_mul(q, so3_exp(dx[6:9])))
            bg = bg + dx[9:12]
            ba = ba + dx[12:15]

            I = np.eye(15)
            P = (I-K@H) @ P @ (I-K@H).T + K @ Rm @ K.T

            up_idx += 1

    event_first3 = []
    for wi, (a, b) in enumerate(windows):
        vals = [nv for nt, nv in zip(nis_times, nis_values) if nt >= b - 1e-12]
        first3 = float(mean(vals[:3])) if vals else float("nan")
        event_first3.append((wi, a, b, first3))

    poserr_rms = float(np.sqrt(np.mean(pos_err*pos_err)))
    return poserr_rms, event_first3, windows


def load_sequence(data_root, method_folder, seq, max_shift_sec=0.2):
    gt_path = data_root / "gt" / seq / "gt_pseudo.csv"
    raw_path = data_root / "raw" / seq / "data.csv"
    den_path = find_corrected_rel(data_root / method_folder, seq)

    gt_ts, gt_data = read_numeric(gt_path, 8)
    gt = GT(gt_ts, gt_data[:, 0:7])

    raw_ts, raw6 = read_numeric(raw_path, 7)
    mask = (raw_ts >= gt.t0_ns) & (raw_ts <= gt.t1_ns)
    raw_ts = raw_ts[mask]
    raw6 = raw6[mask]

    if len(raw_ts) < 1000:
        raise RuntimeError(
            f"Too few raw IMU samples inside GT span: {seq}\n"
            f"Raw timestamp range after crop: {int(raw_ts.min()) if len(raw_ts) else 'none'} .. "
            f"{int(raw_ts.max()) if len(raw_ts) else 'none'}\n"
            f"GT timestamp range: {int(gt.t0_ns)} .. {int(gt.t1_ns)}"
        )

    t_raw = (raw_ts.astype(np.float64) - float(gt.t0_ns)) * 1e-9
    dt_med = float(np.median(np.diff(t_raw)))

    t_den, den6 = read_relative_seconds(den_path, 7)
    den_aligned = interp_to_time(t_raw, t_den, den6)

    shift_samples = 0
    shift_corr = float("nan")
    if max_shift_sec > 0.0 and dt_med > 0.0:
        max_shift = int(round(max_shift_sec / dt_med))
        if max_shift > 0:
            raw_g = np.linalg.norm(raw6[:, 0:3], axis=1)
            den_g = np.linalg.norm(den_aligned[:, 0:3], axis=1)
            shift_samples, shift_corr = estimate_sample_shift(raw_g, den_g, max_shift)
            den_aligned = apply_shift(den_aligned, shift_samples)

    return t_raw, den_aligned, gt, den_path, gt_path, raw_path, shift_samples, shift_corr


def safe_mean(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(mean(vals)) if vals else float("nan")


def safe_std(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    return float(pstdev(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--max_shift_sec", type=float, default=0.2)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    run_rows = []
    event_rows = []
    summary_rows = []

    cache = {}

    for seq in SEQUENCES:
        for method, method_folder in METHODS.items():
            t, imu6, gt, imu_path, gt_path, raw_path, shift_samples, shift_corr = load_sequence(data_root, method_folder, seq, args.max_shift_sec)
            cache[(seq, method)] = (t, imu6, gt)

            print(f"[DATA] {seq} | {method}")
            print(f"       IMU: {imu_path}")
            print(f"       GT : {gt_path}")
            print(f"       RAW: {raw_path}")
            print(f"       lag: {shift_samples} samples, corr={shift_corr:.6f}")

            poserr_by_seed = []
            first3_all = []

            for seed in seeds:
                poserr_rms, event_first3, windows = run_eskf(t, imu6, gt, seed)
                poserr_by_seed.append(poserr_rms)

                run_first3 = [x[3] for x in event_first3 if math.isfinite(x[3])]
                run_rows.append({
                    "sequence": seq,
                    "method": method,
                    "seed": seed,
                    "num_outages": len(windows),
                    "poserr_rms": poserr_rms,
                    "first3_nis_seed_mean": safe_mean(run_first3),
                })

                for wi, a, b, first3 in event_first3:
                    event_rows.append({
                        "sequence": seq,
                        "method": method,
                        "seed": seed,
                        "window_idx": wi,
                        "outage_start_s": a,
                        "outage_end_s": b,
                        "outage_duration_s": b-a,
                        "first3_nis": first3,
                    })
                    if math.isfinite(first3):
                        first3_all.append(first3)

            summary_rows.append({
                "sequence": seq,
                "method": method,
                "num_seeds": len(seeds),
                "num_events": len(first3_all),
                "poserr_rms_mean": safe_mean(poserr_by_seed),
                "poserr_rms_std": safe_std(poserr_by_seed),
                "first3_nis_mean": safe_mean(first3_all),
                "first3_nis_std": safe_std(first3_all),
            })

    run_csv = output_dir / "per_seed_metrics.csv"
    with open(run_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)

    event_csv = output_dir / "per_outage_event_metrics.csv"
    with open(event_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
        w.writeheader()
        w.writerows(event_rows)

    summary_csv = output_dir / "table3_baseline_ours.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    index = {(r["sequence"], r["method"]): r for r in summary_rows}

    print()
    print("PosErr RMS [m]")
    print(f'{"Sequence":<28}{"Baseline":>12}{"Ours":>12}')
    for seq in SEQUENCES:
        b = index[(seq, "Baseline")]["poserr_rms_mean"]
        o = index[(seq, "Ours")]["poserr_rms_mean"]
        print(f"{seq:<28}{b:>12.3f}{o:>12.3f}")

    print()
    print("First-3 NIS [-]")
    print(f'{"Sequence":<28}{"Baseline":>12}{"Ours":>12}')
    for seq in SEQUENCES:
        b = index[(seq, "Baseline")]["first3_nis_mean"]
        o = index[(seq, "Ours")]["first3_nis_mean"]
        print(f"{seq:<28}{b:>12.3f}{o:>12.3f}")

    print()
    print(f"[WRITE] {run_csv}")
    print(f"[WRITE] {event_csv}")
    print(f"[WRITE] {summary_csv}")


if __name__ == "__main__":
    main()
