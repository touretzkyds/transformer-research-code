"""Flatten loss_curves plot layout: move plots out of loss/ and bleu/ subdirs.

Old layout (per experiment):
    .../seq_50/loss/loss_loss.png
    .../seq_50/loss/bleu_bleu.png
    .../seq_50/bleu/          (often empty)

New layout:
    .../seq_50/loss.png
    .../seq_50/bleu.png
    .../seq_50/raw_values/

Usage:
    python utils/visualization/migrate_loss_curve_layout.py --dry-run
    python utils/visualization/migrate_loss_curve_layout.py
"""

import argparse
import shutil
from pathlib import Path

ARTIFACTS_ROOT = Path("artifacts/loss_curves")

RENAME_MAP = {
    "loss_loss": "loss",
    "bleu_bleu": "bleu",
}


def migrate_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if dst.exists():
        if src.stat().st_size == dst.stat().st_size:
            action = f"skip duplicate: {src} (target exists)"
        else:
            action = f"WARN conflict: {src} -> {dst} (target exists, keeping newer/larger)"
            if not dry_run and src.stat().st_mtime >= dst.stat().st_mtime:
                shutil.move(str(src), str(dst))
                return True
        print(f"  {action}")
        return False

    print(f"  move: {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True


def migrate_experiment_dir(exp_dir: Path, dry_run: bool) -> int:
    moved = 0
    for subdir_name in ("loss", "bleu"):
        subdir = exp_dir / subdir_name
        if not subdir.is_dir():
            continue

        for src in sorted(subdir.iterdir()):
            if not src.is_file():
                continue

            stem = src.stem
            if stem in RENAME_MAP:
                dst_name = f"{RENAME_MAP[stem]}{src.suffix}"
            elif stem.startswith("dataset_size_"):
                prefix = "loss" if subdir_name == "loss" else "bleu"
                dst_name = f"{prefix}_{stem}{src.suffix}"
            else:
                dst_name = f"{subdir_name}_{stem}{src.suffix}"

            if migrate_file(src, exp_dir / dst_name, dry_run):
                moved += 1

        if not dry_run:
            try:
                subdir.rmdir()
                print(f"  removed empty dir: {subdir}")
            except OSError:
                remaining = list(subdir.iterdir())
                if remaining:
                    print(f"  kept non-empty dir: {subdir} ({len(remaining)} item(s) left)")

    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files.")
    args = parser.parse_args()

    if not ARTIFACTS_ROOT.is_dir():
        raise SystemExit(f"Not found: {ARTIFACTS_ROOT}")

    experiment_dirs = sorted({p.parent for p in ARTIFACTS_ROOT.rglob("raw_values")})
    print(f"Found {len(experiment_dirs)} experiment dir(s) with raw_values/.\n")

    total_moved = 0
    for exp_dir in experiment_dirs:
        rel = exp_dir.relative_to(ARTIFACTS_ROOT)
        has_nested = (exp_dir / "loss").is_dir() or (exp_dir / "bleu").is_dir()
        if not has_nested:
            continue
        print(f"Processing: {rel}")
        total_moved += migrate_experiment_dir(exp_dir, args.dry_run)

    # Also flatten legacy N1/N6 top-level loss/ and bleu/ subdirs.
    for legacy_dir in sorted(ARTIFACTS_ROOT.glob("N*")):
        if not legacy_dir.is_dir() or legacy_dir.name.startswith("N_"):
            continue
        for subdir_name in ("loss", "bleu"):
            subdir = legacy_dir / subdir_name
            if not subdir.is_dir():
                continue
            for src in sorted(subdir.iterdir()):
                if not src.is_file():
                    continue
                prefix = "loss" if subdir_name == "loss" else "bleu"
                dst_name = f"{prefix}_{src.stem}{src.suffix}"
                print(f"Processing legacy: {src.relative_to(ARTIFACTS_ROOT)}")
                if migrate_file(src, legacy_dir / dst_name, args.dry_run):
                    total_moved += 1
            if not args.dry_run:
                try:
                    subdir.rmdir()
                    print(f"  removed empty dir: {subdir}")
                except OSError:
                    pass

    print(f"\nDone. Moved {total_moved} file(s)." + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
