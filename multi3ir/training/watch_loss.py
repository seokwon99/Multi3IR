"""Re-render loss.png whenever a training log grows. Lines: "[step N] loss=X".

    python -m multi3ir.training.watch_loss results/logs/training/<run>.log [--out <png>] [--once]
"""
from __future__ import annotations
import argparse
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"step (\d+)\] loss=([\d.]+)")


def parse(path: Path):
    """-> {step: loss}; repeated steps (DDP ranks) are averaged."""
    by = {}
    for line in path.read_text(errors="ignore").splitlines():
        if m := PAT.search(line):
            by.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return {s: sum(v) / len(v) for s, v in sorted(by.items())}


def smooth(ys, w=10):
    return [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]


def render(logs, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, path in logs:
        d = parse(path)
        if not d:
            continue
        steps, loss = list(d), list(d.values())
        line, = ax.plot(steps, loss, alpha=0.25, lw=1)
        ax.plot(steps, smooth(loss), label=tag, color=line.get_color(), lw=2)
    ax.set(xlabel="step", ylabel="loss", title="training loss")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png.tmp"), dpi=140, format="png")
    os.replace(out.with_suffix(".png.tmp"), out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None, help="default: <first log>.loss.png")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    logs = [(Path(p).stem, Path(p)) for p in a.logs]
    out = Path(a.out) if a.out else logs[0][1].with_suffix(".loss.png")
    last = None
    while True:
        sig = tuple((p.stat().st_mtime, p.stat().st_size) if p.exists() else (0, 0) for _, p in logs)
        if sig != last:
            render(logs, out)
            print(f"[{time.strftime('%H:%M:%S')}] {out}", flush=True)
            last = sig
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
