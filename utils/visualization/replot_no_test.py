"""Re-generate bleu and loss plots from raw_values/, excluding test metrics.

Walks every raw_values/ directory under artifacts/loss_curves/ that contains
test_bleu.txt (i.e. experiments that originally had the green test line).
For each, it produces new bleu.svg and loss.svg with only train
and val curves, and writes them to a parallel tree under
artifacts/loss_curves_no_test/.

Usage:
    python utils/visualization/replot_no_test.py          # process all
    python utils/visualization/replot_no_test.py --dry-run # list what would be processed
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from matplotlib.ticker import MaxNLocator

ARTIFACTS_ROOT = Path("artifacts/loss_curves")
OUTPUT_ROOT = Path("artifacts/loss_curves_no_test")


def parse_raw_file(path: Path) -> Tuple[List[int], List[float]]:
    """Read a ``step: value`` text file and return (steps, values).

    Handles duplicate lines (val/test files log twice per epoch) and
    files with interleaved runs (train_loss files that were appended to
    across restarts) by keeping only the last contiguous run.
    """
    steps, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, v = line.split(":", 1)
            steps.append(int(s))
            values.append(float(v))

    if not steps:
        return [], []

    # Detect multiple runs: if the step ever decreases, keep only from
    # the last reset onward.
    last_reset = 0
    for i in range(1, len(steps)):
        if steps[i] < steps[i - 1]:
            last_reset = i
    steps = steps[last_reset:]
    values = values[last_reset:]

    # Deduplicate (val/test files sometimes have each epoch twice)
    seen = {}
    for s, v in zip(steps, values):
        seen[s] = v
    deduped = sorted(seen.items())
    steps = [s for s, _ in deduped]
    values = [v for _, v in deduped]

    return steps, values


def interpolate(array: List[float], target_length: int) -> List[float]:
    if len(array) == 0:
        return []
    if len(array) == 1:
        return array * target_length
    mesh = interp.interp1d(np.arange(len(array)), array)
    return mesh(np.linspace(0, len(array) - 1, target_length)).tolist()


UPPERCASE_KEYS = {"bs", "n"}

def build_title(rel_path: str) -> str:
    """Build a title string like 'N: 6 | Name: wmt24 | ...' from the
    directory path components (plot type is omitted; the y-axis conveys it)."""
    parts = Path(rel_path).parts  # e.g. N_6, name_wmt24, size_3M, ...
    segments = []
    for part in parts:
        if "_" in part:
            key, _, val = part.partition("_")
            display_key = key.upper() if key.lower() in UPPERCASE_KEYS else key.capitalize()
            segments.append(f"{display_key}: {val}")
    return " | ".join(segments)


LABEL_MAP = {
    "train_bleu": "Train BLEU",
    "val_bleu":   "Val BLEU",
    "train_loss": "Train loss",
    "val_loss":   "Val loss",
}

VAL_METRICS = {"val_bleu", "val_loss"}

def make_plot(
    metric_series: Dict[str, List[float]],
    plot_type: str,
    title: str,
    output_path: Path,
):
    if plot_type == "loss":
        ylim = (0, 9)
        ylabel = "Loss"
    else:
        ylim = (0, 1)
        ylabel = "BLEU"

    max_length = max(len(v) for v in metric_series.values())

    fig, ax = plt.subplots(dpi=300)
    for name, values in metric_series.items():
        label = LABEL_MAP.get(name, name.replace("_", " ").capitalize())
        series = interpolate(values, max_length)
        kwargs = {}
        if name in VAL_METRICS:
            kwargs["marker"] = "o"
            kwargs["markersize"] = 3
        ax.plot(range(1, len(series) + 1), series, label=label, **kwargs)

    ax.set_ylim(ylim)
    ax.set_xlim(1, max_length)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Weight Updates")
    ax.set_ylabel(ylabel)
    ax.set_title(title, y=1.15)

    # secondary epoch axis
    for name in ("val_bleu", "val_loss"):
        if name in metric_series:
            n_epochs = len(metric_series[name])
            break
    else:
        n_epochs = None

    if n_epochs and n_epochs > 0:
        ax2 = ax.twiny()
        ax2.set_xlim(1, n_epochs)
        ax2.set_xlabel("Epoch")

    ax.grid(visible=True)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format="svg", bbox_inches="tight")
    plt.close(fig)


def process_experiment(raw_dir: Path, rel_path: str, dry_run: bool = False):
    """Regenerate bleu and loss plots for one experiment."""
    out_base = OUTPUT_ROOT / rel_path / "loss"

    plots = [
        ("bleu", ["train_bleu", "val_bleu"], "bleu.svg"),
        ("loss", ["train_loss", "val_loss"], "loss.svg"),
    ]

    for plot_type, metric_names, filename in plots:
        series: Dict[str, List[float]] = {}
        for name in metric_names:
            fpath = raw_dir / f"{name}.txt"
            if not fpath.exists():
                continue
            _, values = parse_raw_file(fpath)
            if values:
                series[name] = values

        if not series:
            continue

        out_path = out_base / filename
        if dry_run:
            print(f"  [dry-run] would write: {out_path}")
            continue

        title = build_title(rel_path)
        make_plot(series, plot_type, title, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list what would be processed, don't write.")
    args = parser.parse_args()

    candidates = sorted(ARTIFACTS_ROOT.rglob("raw_values/test_bleu.txt"))
    print(f"Found {len(candidates)} experiment(s) with test data.\n")

    for test_file in candidates:
        raw_dir = test_file.parent
        # rel_path: e.g. N_6/name_wmt24/size_3M/bs_128/seed_42/seq_50
        experiment_dir = raw_dir.parent
        rel_path = str(experiment_dir.relative_to(ARTIFACTS_ROOT))
        print(f"Processing: {rel_path}")
        process_experiment(raw_dir, rel_path, dry_run=args.dry_run)

    print(f"\nDone. Output in: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
