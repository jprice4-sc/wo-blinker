#!/usr/bin/env python3
"""Time-trace blinking analysis for PL video of a microLED panel.

Pipeline (as used on 50x_50ms_frame_10fps.avi):
  1. Stream the video once: temporal mean/std maps, global-mean time series,
     and a downsampled stack for per-site time series (memory-safe).
  2. Grid the field into one-site-per-LED blocks.
  3. Detect telegraph switchers with a split-median step statistic.
  4. Extract ON/OFF dwell times by mid-level thresholding.
  5. Poisson trap density, blinker/brightness correlation, NN clustering.
  6. Stretched-exponential fit of the global photo-activation transient.
  7. Figures: overview, global statistics, per-blinker traces.

Usage:
    python pl_blinking_analysis.py video.avi [--outdir out] [--fps 10]
        [--block 6] [--ds 512] [--rel-range 0.08] [--step-thresh 0.04]

Dependencies: numpy, scipy, opencv-python, matplotlib
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# 1. Streaming load
# ----------------------------------------------------------------------
def stream_video(path, ds_size=512):
    """One pass over the video. Returns per-pixel mean/std maps (full res),
    global mean per frame, and a downsampled float16 stack (T, ds, ds).
    """
    cap = cv2.VideoCapture(path)
    s = s2 = None
    n = 0
    gmeans, ds_frames = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if s is None:
            s, s2 = np.zeros_like(g), np.zeros_like(g)
        s += g
        s2 += g * g
        n += 1
        gmeans.append(g.mean())
        ds_frames.append(cv2.resize(g, (ds_size, ds_size), interpolation=cv2.INTER_AREA).astype(np.float16))
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames read from {path}")
    mean_map = s / n
    std_map = np.sqrt(np.clip(s2 / n - mean_map**2, 0, None))
    return mean_map, std_map, np.asarray(gmeans), np.stack(ds_frames)


# ----------------------------------------------------------------------
# 2-3. Site grid + telegraph detection
# ----------------------------------------------------------------------
def site_timeseries(ds_stack, block=6):
    """Average the downsampled stack into (T, hb, wb) site blocks
    (~one LED per block) and remove the global drift.
    """
    T, H, W = ds_stack.shape
    hb, wb = H // block, W // block
    blk = ds_stack[:, : hb * block, : wb * block].astype(np.float32).reshape(T, hb, block, wb, block).mean(axis=(2, 4))
    g = blk.mean(axis=(1, 2), keepdims=True)
    return blk - g + g.mean()  # detrended, global level preserved


def step_score(ts, tmin=25, tmax=None, stride=10):
    """Max |median(first part) - median(second part)| over split points.
    Robust two-level (telegraph) step detector.
    """
    tmax = tmax if tmax is not None else len(ts) - tmin
    best = 0.0
    for k in range(tmin, tmax, stride):
        d = abs(np.median(ts[:k]) - np.median(ts[k:]))
        best = max(best, d)
    return best


def detect_blinkers(blk, rel_range=0.08, step_thresh=0.04):
    """Return boolean map of telegraph switchers and their step amplitudes."""
    mn = blk.mean(0)
    rel = (blk.max(0) - blk.min(0)) / (mn + 1)
    blink = np.zeros(mn.shape, bool)
    amp = np.zeros(mn.shape)
    for i, j in zip(*np.where(rel > rel_range), strict=False):
        a = step_score(blk[:, i, j]) / (mn[i, j] + 1)
        if a > step_thresh:
            blink[i, j] = True
            amp[i, j] = a
    return blink, amp, mn


# ----------------------------------------------------------------------
# 4. Dwell times
# ----------------------------------------------------------------------
def dwell_times(blk, blink, fps):
    """Mid-level threshold each blinker; return ON/OFF dwell arrays (seconds)."""
    on, off = [], []
    for i, j in zip(*np.where(blink), strict=False):
        ts = blk[:, i, j]
        lo, hi = np.percentile(ts, 10), np.percentile(ts, 90)
        if hi - lo < 3:  # amplitude below noise floor for thresholding
            continue
        state = ts > (lo + hi) / 2
        edges = np.where(np.concatenate(([1], np.diff(state.astype(int)) != 0, [1])))[0]
        runs = np.diff(edges)
        first = state[0]
        for k, r in enumerate(runs):
            (on if (first ^ (k % 2 == 1)) else off).append(r / fps)
    return np.asarray(on), np.asarray(off)


# ----------------------------------------------------------------------
# 5. Population statistics
# ----------------------------------------------------------------------
def population_stats(blink, mn):
    N = blink.size
    n = int(blink.sum())
    lam = -np.log(1 - n / N) if n < N else np.inf  # Poisson mean traps/pixel
    dim = mn < np.median(mn)
    rate_dim, rate_bright = blink[dim].mean(), blink[~dim].mean()
    pts = np.stack(np.where(blink), 1).astype(float)
    nn_ratio = np.nan
    if len(pts) > 3:
        d, _ = cKDTree(pts).query(pts, k=2)
        nn_ratio = d[:, 1].mean() / (0.5 / np.sqrt(n / N))  # <1 => clustered
    return dict(n=n, N=N, lam=lam, rate_dim=rate_dim, rate_bright=rate_bright, nn_ratio=nn_ratio)


# ----------------------------------------------------------------------
# 6. Global transient
# ----------------------------------------------------------------------
def fit_transient(gmeans, fps):
    """Stretched exponential I(t) = I_inf - dI * exp(-(t/tau)^beta)."""
    t = np.arange(len(gmeans)) / fps

    def f(t, Iinf, dI, tau, beta):
        return Iinf - dI * np.exp(-((t / tau) ** np.clip(beta, 0.05, 2)))

    p0 = [gmeans[-1], max(gmeans[-1] - gmeans[0], 1e-3), 3.0, 0.7]
    try:
        popt, _ = optimize.curve_fit(f, t, gmeans, p0=p0, maxfev=20000)
        return dict(I_inf=popt[0], dI=popt[1], tau=popt[2], beta=popt[3])
    except RuntimeError:
        return None


# ----------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------
def make_figures(outdir, mean_map, gmeans, blk, blink, amp, mn, on, off, fps, fit):
    t = np.arange(blk.shape[0]) / fps

    # --- overview ---
    cvm = blk.std(0) / (mn + 1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    im = ax[0, 0].imshow(mean_map[::4, ::4], cmap="gray")
    ax[0, 0].set_title("Temporal mean")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046)
    im = ax[0, 1].imshow(cvm, cmap="inferno", vmax=np.percentile(cvm, 99.5))
    ys, xs = np.where(blink)
    ax[0, 1].plot(xs, ys, "o", mfc="none", mec="cyan", ms=5, mew=0.6)
    ax[0, 1].set_title(f"Fluctuation map, {blink.sum()} blinkers circled")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[1, 0].plot(np.arange(len(gmeans)) / fps, gmeans)
    if fit:
        tf = np.arange(len(gmeans)) / fps
        ax[1, 0].plot(
            tf,
            fit["I_inf"] - fit["dI"] * np.exp(-((tf / fit["tau"]) ** fit["beta"])),
            "--",
            label=f"stretched exp: tau={fit['tau']:.1f}s beta={fit['beta']:.2f}",
        )
        ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_title("Global mean vs time")
    order = np.argsort(amp[ys, xs])[::-1][:6]
    for k in order:
        ax[1, 1].plot(t, blk[:, ys[k], xs[k]], lw=0.8, label=f"({ys[k]},{xs[k]})")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_title("Strongest blinker traces")
    ax[1, 1].legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "overview.png"), dpi=100)
    plt.close(fig)

    # --- global statistics ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    rel = mn / np.median(mn)
    ax[0].hist(rel.ravel(), bins=120)
    ax[0].set_title("Per-pixel relative IQE")
    ax[0].set_xlabel("I / median")
    m = mn - mn.mean()
    F = np.fft.fft2(m)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    ac /= ac.max()
    cy, cx = np.array(ac.shape) // 2
    ax[1].plot(ac[cy, cx : cx + 30], "-o", ms=3)
    ax[1].axhline(1 / np.e, ls="--", c="gray")
    ax[1].set_title("Spatial autocorrelation of mottle")
    ax[1].set_xlabel("lag (site pitches)")
    bins = np.logspace(-1, 1.5, 15)
    for arr, lab in [(on, "ON"), (off, "OFF")]:
        if len(arr):
            h, e = np.histogram(arr, bins=bins)
            ax[2].loglog(np.sqrt(e[1:] * e[:-1]), h + 0.5, "o-", label=lab)
    ax[2].set_title("Dwell-time distributions")
    ax[2].set_xlabel("dwell (s)")
    ax[2].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "global_stats.png"), dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--outdir", default="pl_analysis_out")
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate (Hz)")
    ap.add_argument("--ds", type=int, default=512, help="downsampled stack size")
    ap.add_argument("--block", type=int, default=6, help="block size in ds pixels (~1 LED)")
    ap.add_argument("--rel-range", type=float, default=0.08, help="candidate threshold (max-min)/mean")
    ap.add_argument("--step-thresh", type=float, default=0.04, help="telegraph step threshold (fractional)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/5] streaming {args.video} ...")
    mean_map, std_map, gmeans, ds = stream_video(args.video, args.ds)
    print(f"      {len(gmeans)} frames, full-res {mean_map.shape}, ds stack {ds.shape}")

    print("[2/5] building site time series ...")
    blk = site_timeseries(ds, args.block)

    print("[3/5] detecting telegraph switchers ...")
    blink, amp, mn = detect_blinkers(blk, args.rel_range, args.step_thresh)
    stats = population_stats(blink, mn)
    print(
        f"      blinkers: {stats['n']}/{stats['N']} = {stats['n'] / stats['N']:.3f}"
        f" | Poisson lambda = {stats['lam']:.4f} traps/pixel"
    )
    print(
        f"      blinker rate dim/bright half: {stats['rate_dim']:.3f}/{stats['rate_bright']:.3f}"
        f" | NN clustering ratio: {stats['nn_ratio']:.2f} (<1 = clustered)"
    )
    if stats["n"]:
        a = amp[blink]
        print(f"      step amplitude: median {np.median(a):.2%}, max {a.max():.2%}")

    print("[4/5] dwell times & transient fit ...")
    on, off = dwell_times(blk, blink, args.fps)
    if len(on):
        print(
            f"      dwell ON:  n={len(on)}  median {np.median(on):.2f}s  mean {on.mean():.2f}s\n"
            f"      dwell OFF: n={len(off)} median {np.median(off):.2f}s  mean {off.mean():.2f}s"
        )
    fit = fit_transient(gmeans, args.fps)
    if fit:
        print(
            f"      transient: dI={fit['dI']:.2f} ({100 * fit['dI'] / fit['I_inf']:.1f}%)"
            f" tau={fit['tau']:.1f}s beta={fit['beta']:.2f}"
        )

    print("[5/5] figures & data ...")
    make_figures(args.outdir, mean_map, gmeans, blk, blink, amp, mn, on, off, args.fps, fit)
    np.savez_compressed(
        os.path.join(args.outdir, "results.npz"),
        mean_map=mean_map.astype(np.float32),
        std_map=std_map.astype(np.float32),
        gmeans=gmeans,
        site_traces=blk.astype(np.float32),
        blink_map=blink,
        blink_amp=amp.astype(np.float32),
        dwell_on=on,
        dwell_off=off,
    )
    print(f"done -> {args.outdir}/overview.png, global_stats.png, results.npz")


if __name__ == "__main__":
    main()
