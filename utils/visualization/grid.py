import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ── Edit this directly to set your default grid ──────────────────────────
# sizes = ["30K", "100K", "300K", "1M", "3M", "5M"]
sizes = ["100K", "300K", "5M"]
DEFAULT_PATHS = [
    [f"artifacts/loss_curves/N_1/name_wmt24/size_{size}/bs_128/seed_42/seq_50/loss/loss_loss.png" for size in sizes],
    [f"artifacts/loss_curves/N_6/name_wmt24/size_{size}/bs_128/seed_42/seq_50/loss/loss_loss.png" for size in sizes],
]
DEFAULT_OUTPUT_DIR = "artifacts"
DEFAULT_OUTPUT_STEM = "grid"
DEFAULT_TITLE = None
# ─────────────────────────────────────────────────────────────────────────


def plot_image_grid(
    paths: Union[List[str], List[List[str]]],
    output_path: str = "artifacts/grid.png",
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    cell_size: Tuple[float, float] = (5, 4),
    dpi: int = 200,
    title: Optional[str] = None,
    labels: Optional[Union[List[str], List[List[str]]]] = None,
):
    """Arrange image files into a tight grid and save the result.

    Parameters
    ----------
    paths : list of str  OR  list of list of str
        Flat list  – images are laid out left-to-right, top-to-bottom
        (auto-grid or controlled by *ncols*/*nrows*).

        Nested list – each inner list is one row, e.g.
        ``[[row1_img1, row1_img2], [row2_img1, row2_img2]]``.
        Rows may have different lengths; shorter rows get blank cells.
    output_path : str
        Where to save the combined grid image.
    ncols : int, optional
        Number of columns (only used with flat *paths*).
    nrows : int, optional
        Number of rows (only used with flat *paths*).
    cell_size : (width, height)
        Size of each subplot in inches.
    dpi : int
        Resolution of the saved figure.
    title : str, optional
        Suptitle for the whole grid.
    labels : list of str  OR  list of list of str, optional
        Per-image labels (must match shape of *paths*).
        Defaults to the filename stem of each path.
    """
    if isinstance(paths[0], list):
        rows: List[List[str]] = paths
        nrows = len(rows)
        ncols = max(len(r) for r in rows)
        label_rows: Optional[List[List[Optional[str]]]] = None
        if labels is not None:
            label_rows = labels
    else:
        flat: List[str] = paths
        n = len(flat)
        if n == 0:
            raise ValueError("No image paths provided.")
        if ncols is None and nrows is None:
            ncols = math.ceil(math.sqrt(n))
        if ncols is None:
            ncols = math.ceil(n / nrows)
        if nrows is None:
            nrows = math.ceil(n / ncols)
        rows = [flat[i * ncols:(i + 1) * ncols] for i in range(nrows)]
        label_rows = None
        if labels is not None:
            label_rows = [labels[i * ncols:(i + 1) * ncols] for i in range(nrows)]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_size[0], nrows * cell_size[1]),
        squeeze=False,
    )

    total = 0
    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            if c < len(row):
                img = mpimg.imread(row[c])
                ax.imshow(img)
                if label_rows and c < len(label_rows[r]):
                    lbl = label_rows[r][c]
                elif label_rows is None:
                    lbl = Path(row[c]).stem
                else:
                    lbl = None
                if lbl:
                    ax.set_title(lbl, fontsize=8)
                total += 1
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    if title:
        fig.subplots_adjust(top=0.94)

    out = Path(output_path)
    suffix = f"_{nrows}x{ncols}"
    if not out.stem.endswith(suffix):
        out = out.with_stem(out.stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid ({nrows}x{ncols}, {total} images) → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple plot images into a tight grid."
    )
    parser.add_argument(
        "images", nargs="*", default=None, help="Paths to image files (omit to use DEFAULT_PATHS)."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: <DEFAULT_OUTPUT_DIR>/<DEFAULT_OUTPUT_STEM>_RxC.png)",
    )
    parser.add_argument("--ncols", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--cell-width", type=float, default=5)
    parser.add_argument("--cell-height", type=float, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    paths = args.images if args.images else DEFAULT_PATHS
    if not paths:
        parser.error("No images provided and DEFAULT_PATHS is empty. "
                      "Either pass image paths as arguments or fill in DEFAULT_PATHS in grid.py.")

    plot_image_grid(
        paths=paths,
        output_path=args.output or str(Path(DEFAULT_OUTPUT_DIR) / f"{DEFAULT_OUTPUT_STEM}.png"),
        ncols=args.ncols,
        nrows=args.nrows,
        cell_size=(args.cell_width, args.cell_height),
        dpi=args.dpi,
        title=args.title or DEFAULT_TITLE,
    )
