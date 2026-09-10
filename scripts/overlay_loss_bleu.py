"""Overlay train/val loss + BLEU, including experiment grids.

Examples:
    python scripts/overlay_loss_bleu.py              # quick single-plot demo
    python scripts/overlay_loss_bleu.py --grid        # full 2x6 from raw_values
    python scripts/overlay_loss_bleu.py --grid-n 1    # 2x3 grid for N=1 only
    python scripts/overlay_loss_bleu.py --grid-split  # 2x3 grids for N=1 and N=6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── Quick-test demo values ────────────────────────────────────────────────
EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
TRAIN_LOSS = [
    7.82, 5.41, 3.95, 2.88, 2.21, 1.74, 1.42, 1.18, 1.01, 0.89,
    0.72, 0.61, 0.54, 0.49, 0.46,
]
VAL_LOSS = [
    6.10, 4.20, 3.05, 2.45, 2.10, 1.88, 1.72, 1.62, 1.55, 1.52,
    1.58, 1.68, 1.80, 1.95, 2.10,
]
VAL_BLEU = [
    0.02, 0.06, 0.11, 0.18, 0.24, 0.31, 0.36, 0.40, 0.43, 0.45,
    0.48, 0.50, 0.51, 0.52, 0.52,
]
TRAIN_BLEU = [
    0.03, 0.09, 0.15, 0.23, 0.30, 0.38, 0.44, 0.49, 0.53, 0.56,
    0.60, 0.63, 0.65, 0.66, 0.67,
]

# ── 2x6 grid config ───────────────────────────────────────────────────────
SIZES = ["30K", "100K", "300K", "1M", "3M", "5M"]
NS = [1, 6]
RAW_ROOTS = [
    Path("artifacts_copy_for_cursor/loss_curves"),
    Path("artifacts_preserved_latest/loss_curves"),
]
GRID_OUTPUT = Path("artifacts_copy_for_cursor/grid_2x6.svg")
GRID_N_OUTPUT = Path("artifacts_copy_for_cursor/grid_N{n}_2x3.svg")
DEMO_OUTPUT = Path("artifacts/overlay_loss_bleu.png")

LOSS_COLOR = "#1f77b4"
BLEU_COLOR = "#d95f02"


def parse_metric_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse `step: value` lines; keep last value per step (handles duplicates)."""
    by_step: Dict[int, float] = {}
    order: List[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        step_s, val_s = line.split(":", 1)
        step = int(step_s.strip())
        val = float(val_s.strip())
        if step not in by_step:
            order.append(step)
        by_step[step] = val
    steps = np.asarray(order, dtype=float)
    vals = np.asarray([by_step[s] for s in order], dtype=float)
    return steps, vals


def load_series_as_list(path: Path) -> np.ndarray:
    """Load values in file order (for dense train_loss logged every update)."""
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals.append(float(line.split(":", 1)[1].strip()))
    return np.asarray(vals, dtype=float)


def find_raw_dir(n: int, size: str) -> Path:
    rel = Path(f"N_{n}/name_wmt24/size_{size}/bs_128/seed_42/seq_50/raw_values")
    for root in RAW_ROOTS:
        candidate = root / rel
        if (candidate / "train_loss.txt").exists() and (candidate / "val_loss.txt").exists():
            return candidate
    raise FileNotFoundError(f"No raw_values for N={n} size={size}")


def load_run(n: int, size: str) -> dict:
    raw = find_raw_dir(n, size)
    train_loss = load_series_as_list(raw / "train_loss.txt")
    _, val_loss = parse_metric_file(raw / "val_loss.txt")
    _, train_bleu = parse_metric_file(raw / "train_bleu.txt")
    _, val_bleu = parse_metric_file(raw / "val_bleu.txt")
    n_epochs = max(len(val_loss), len(train_bleu), len(val_bleu), 1)
    return {
        "n": n,
        "size": size,
        "raw": raw,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_bleu": train_bleu,
        "val_bleu": val_bleu,
        "n_epochs": n_epochs,
    }


def downsample(y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_points:
        x = np.arange(1, len(y) + 1, dtype=float)
        return x, y
    idx = np.linspace(0, len(y) - 1, max_points).astype(int)
    x = (idx + 1).astype(float)
    return x, y[idx]


def stretch_to_updates(y: np.ndarray, n_updates: int) -> np.ndarray:
    """Linearly map a per-epoch series onto the weight-update axis."""
    if len(y) == 0:
        return np.full(n_updates, np.nan)
    if len(y) == 1:
        return np.full(n_updates, y[0])
    xp = np.linspace(0, n_updates - 1, len(y))
    x = np.arange(n_updates, dtype=float)
    return np.interp(x, xp, y)


def style_loss_axis(ax):
    ax.set_ylabel("Loss", color=LOSS_COLOR, fontweight="semibold")
    ax.tick_params(axis="y", colors=LOSS_COLOR, labelsize=7)
    ax.spines["left"].set_color(LOSS_COLOR)
    ax.set_ylim(0, 9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)


def style_bleu_axis(ax):
    ax.set_ylabel("BLEU", color=BLEU_COLOR, fontweight="semibold")
    ax.tick_params(axis="y", colors=BLEU_COLOR, labelsize=7)
    ax.spines["right"].set_color(BLEU_COLOR)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(0, 1)


def plot_overlay_on_ax(
    ax_loss,
    run: dict,
    max_points: int = 2500,
    show_legend: bool = False,
    show_ylabels: bool = True,
):
    train_loss = run["train_loss"]
    n_updates = len(train_loss)
    x_loss, y_train = downsample(train_loss, max_points)

    y_val = stretch_to_updates(run["val_loss"], n_updates)
    y_train_bleu = stretch_to_updates(run["train_bleu"], n_updates)
    y_val_bleu = stretch_to_updates(run["val_bleu"], n_updates)
    _, y_val = downsample(y_val, max_points)
    _, y_train_bleu = downsample(y_train_bleu, max_points)
    _, y_val_bleu = downsample(y_val_bleu, max_points)

    ax_bleu = ax_loss.twinx()

    (ln1,) = ax_loss.plot(x_loss, y_train, color=LOSS_COLOR, lw=1.2, label="Train loss", zorder=3)
    (ln2,) = ax_loss.plot(x_loss, y_val, color=LOSS_COLOR, lw=1.2, ls="--", label="Val loss", zorder=4)
    (ln3,) = ax_bleu.plot(x_loss, y_train_bleu, color=BLEU_COLOR, lw=1.2, label="Train BLEU", zorder=3)
    (ln4,) = ax_bleu.plot(
        x_loss, y_val_bleu, color=BLEU_COLOR, lw=1.2, ls="--", label="Val BLEU", zorder=4
    )

    ax_loss.set_xlim(1, n_updates)
    ax_loss.set_xlabel("Weight Updates", fontsize=8)
    ax_loss.tick_params(axis="x", labelsize=7)
    style_loss_axis(ax_loss)
    style_bleu_axis(ax_bleu)
    if not show_ylabels:
        ax_loss.set_ylabel("")
        ax_bleu.set_ylabel("")

    # Epoch secondary axis
    ax_ep = ax_loss.twiny()
    ax_ep.set_xlim(0, run["n_epochs"])
    ax_ep.set_xlabel("Epoch", fontsize=8)
    ax_ep.tick_params(axis="x", labelsize=7)
    ax_ep.spines["top"].set_visible(True)

    title = f"N: {run['n']} | Size: {run['size']} | Bs: 128 | Seed: 42"
    ax_loss.set_title(title, fontsize=8, pad=18)

    if show_legend:
        lines = [ln1, ln2, ln3, ln4]
        ax_loss.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=6, framealpha=0.9)

    return ax_bleu


def overlay_loss_bleu(
    epochs,
    train_loss,
    val_loss,
    val_bleu,
    train_bleu=None,
    output_path: Path = DEMO_OUTPUT,
    title: str = "Loss & BLEU",
):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax_bleu = ax.twinx()
    x = np.asarray(epochs, dtype=float)
    (ln1,) = ax.plot(x, train_loss, color=LOSS_COLOR, lw=2.2, marker="o", ms=4, label="Train loss")
    (ln2,) = ax.plot(x, val_loss, color=LOSS_COLOR, lw=2.2, ls="--", marker="D", ms=4, label="Val loss")
    (ln3,) = ax_bleu.plot(x, val_bleu, color=BLEU_COLOR, lw=2.2, ls="--", marker="s", ms=4, label="Val BLEU")
    lines = [ln1, ln2, ln3]
    if train_bleu is not None:
        (ln4,) = ax_bleu.plot(
            x, train_bleu, color=BLEU_COLOR, lw=2.2, marker="^", ms=4, label="Train BLEU"
        )
        lines.append(ln4)
    ax.set_xlabel("Epoch")
    style_loss_axis(ax)
    style_bleu_axis(ax_bleu)
    ax.set_ylim(bottom=0)
    ax.set_title(title, pad=10)
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", framealpha=0.9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def generate_grid_2x3_for_n(
    n: int,
    sizes: Sequence[str] = SIZES,
    output_path: Optional[Path] = None,
    dpi: int = 160,
):
    """Build a 2x3 overlay grid for one layer count across all dataset sizes."""
    if len(sizes) != 6:
        raise ValueError(f"Expected 6 sizes for a 2x3 grid, got {len(sizes)}")
    size_rows = [list(sizes[:3]), list(sizes[3:])]
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.2, nrows * 3.4),
        dpi=dpi,
        squeeze=False,
    )

    for r, row_sizes in enumerate(size_rows):
        for c, size in enumerate(row_sizes):
            ax = axes[r][c]
            run = load_run(n, size)
            print(f"Plotting N={n} size={size} from {run['raw']}")
            plot_overlay_on_ax(
                ax,
                run,
                show_legend=(r == 0 and c == ncols - 1),
                show_ylabels=(c == 0),
            )

    fig.suptitle(f"Loss & BLEU overlay | N={n} | wmt24 | seq 50", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=1.2, h_pad=1.4)

    out = Path(output_path or GRID_N_OUTPUT.as_posix().format(n=n))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid ({nrows}x{ncols}, N={n}) → {out}")
    return out


def generate_grid_split_by_n(
    ns: Sequence[int] = NS,
    sizes: Sequence[str] = SIZES,
    output_dir: Path = GRID_N_OUTPUT.parent,
    dpi: int = 160,
) -> List[Path]:
    """Build one 2x3 grid per layer count."""
    paths = []
    for n in ns:
        out = output_dir / f"grid_N{n}_2x3.svg"
        paths.append(generate_grid_2x3_for_n(n=n, sizes=sizes, output_path=out, dpi=dpi))
    return paths


def generate_grid_2x6(
    sizes: Sequence[str] = SIZES,
    ns: Sequence[int] = NS,
    output_path: Path = GRID_OUTPUT,
    dpi: int = 160,
):
    nrows, ncols = len(ns), len(sizes)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.2, nrows * 3.4),
        dpi=dpi,
        squeeze=False,
    )

    for r, n in enumerate(ns):
        for c, size in enumerate(sizes):
            ax = axes[r][c]
            run = load_run(n, size)
            print(f"Plotting N={n} size={size} from {run['raw']}")
            plot_overlay_on_ax(
                ax,
                run,
                show_legend=(r == 0 and c == ncols - 1),
                show_ylabels=(c == 0),
            )

    fig.suptitle("Loss & BLEU overlay | wmt24 | seq 50", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=1.2, h_pad=1.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid ({nrows}x{ncols}) → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true", help="Build the full 2x6 overlay grid from raw_values")
    parser.add_argument(
        "--grid-n",
        type=int,
        choices=NS,
        default=None,
        help="Build a 2x3 overlay grid for a single layer count (1 or 6)",
    )
    parser.add_argument(
        "--grid-split",
        action="store_true",
        help="Build separate 2x3 grids for each layer count (N=1 and N=6)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default depends on mode)",
    )
    args = parser.parse_args()

    if args.grid_split:
        generate_grid_split_by_n()
    elif args.grid_n is not None:
        generate_grid_2x3_for_n(n=args.grid_n, output_path=args.output)
    elif args.grid:
        generate_grid_2x6(output_path=args.output or GRID_OUTPUT)
    else:
        overlay_loss_bleu(
            EPOCHS,
            TRAIN_LOSS,
            VAL_LOSS,
            VAL_BLEU,
            train_bleu=TRAIN_BLEU,
            output_path=args.output or DEMO_OUTPUT,
        )


if __name__ == "__main__":
    main()
