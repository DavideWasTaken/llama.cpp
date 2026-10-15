#!/usr/bin/env python3
# Charts of docs/moe-cache-hybrid.md, in a light and a dark version for GitHub:
# - expert use per layer, from a --moe-cache-profile file
# - generation speed, from the measured results below
#
# usage: python scripts/moe-cache-hybrid-plots.py <profile> [out_dir]

import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "base": "#c3c2b7", "accent": "#2a78d6", "context": "#c3c2b7",
              "ramp": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
              "base": "#383835", "accent": "#3987e5", "context": "#5f5e5a",
              "ramp": ["#1a1a19", "#104281", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb"]},
}

# measured on the test system of docs/moe-cache-hybrid.md (t/s), each pair in the same session
SPEED = [
    ("llama-server, MTP draft", 38.0, 49.7),
    ("no speculative decoding", 34.1, 43.0),
]


def read_profile(path):
    with open(path, "rb") as f:
        magic, version, n_layer, n_expert = struct.unpack("<4I", f.read(16))
        if magic != 0x50434d4c or version != 1:
            raise SystemExit(f"{path}: not a MoE cache profile")
        scores = np.frombuffer(f.read(), dtype=np.float32).reshape(n_layer, n_expert)
    return scores[scores.sum(axis=1) > 0]  # the layers without data (e.g. MTP) are skipped


def setup(t):
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["base"],
    })


def plot_experts(scores, t, out, cache_frac):
    share = scores / scores.sum(axis=1, keepdims=True)
    share = -np.sort(-share, axis=1)  # each layer from its most used expert to its least used
    n_layer, n_expert = share.shape
    k = int(round(cache_frac * n_expert))
    covered = np.median(share[:, :k].sum(axis=1))

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="none")
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    im = ax.imshow(share, aspect="auto", cmap=cmap, norm=PowerNorm(0.4, vmin=0, vmax=share.max()),
                   interpolation="nearest", extent=(0, n_expert, n_layer, 0))
    ax.axvline(k, color=t["ink"], linewidth=1.5)
    ax.text(k + 4, 3, f"VRAM cache: {cache_frac:.0%} of the experts\n= {covered:.0%} of their uses (median layer)",
            color=t["ink"], va="top", fontsize=10)
    ax.set_xlabel("experts of the layer, from most to least used (256)")
    ax.set_ylabel("MoE layer")
    ax.set_yticks([0, 10, 20, 30, 39])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.outline.set_visible(False)
    cb.set_label("share of the layer's expert uses", color=t["ink2"])
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Few experts do most of the work", loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=10)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_speed(t, out):
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor="none")
    h = 0.34
    for i, (name, base, fork) in enumerate(SPEED):
        y = i
        ax.barh(y - h/2 - 0.02, base, height=h, color=t["context"], label="llama.cpp, fastest setup without the cache" if i == 0 else None)
        ax.barh(y + h/2 + 0.02, fork, height=h, color=t["accent"], label="this fork, hybrid cache + profile" if i == 0 else None)
        ax.text(base + 0.6, y - h/2 - 0.02, f"{base:.1f}", va="center", color=t["ink2"])
        ax.text(fork + 0.6, y + h/2 + 0.02, f"{fork:.1f}  (+{100*(fork/base - 1):.0f}%)", va="center", color=t["ink"], fontweight="bold")
    ax.set_yticks(range(len(SPEED)), [s[0] for s in SPEED], color=t["ink2"])
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("token generation (t/s), Qwen3.6-35B-A3B on an RTX 5060 Ti 8 GB, PCIe 3.0 x8")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=t["ink2"],
              handlelength=1.2, borderaxespad=0.2)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    args = [a for a in sys.argv[1:] if a != "--png"]
    ext  = "png" if "--png" in sys.argv else "svg"  # png: preview
    if not args:
        raise SystemExit("usage: moe-cache-hybrid-plots.py <profile> [out_dir] [--png]")
    scores = read_profile(args[0])
    out_dir = Path(args[1]) if len(args) > 1 else Path(__file__).parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_frac = 0.17  # 3200 MiB of cache = 17% of the experts of Qwen3.6-35B-A3B UD-Q4_K_M
    for name, t in THEMES.items():
        setup(t)
        plot_experts(scores, t, out_dir / f"moe-cache-hybrid-experts-{name}.{ext}", cache_frac)
        plot_speed(t, out_dir / f"moe-cache-hybrid-speed-{name}.{ext}")
    print(f"charts written to {out_dir}")


if __name__ == "__main__":
    main()
