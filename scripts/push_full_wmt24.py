#!/usr/bin/env python3
"""Push full WMT24 de-en config to Hub, with corrected val/test splits."""

import os
import os.path as op
import sys

sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))

from datasets import DatasetDict, load_from_disk

FULL_PATH = "artifacts/saved_data/tokenized_data/wmt24_de_en_reformatted_new"
GOOD_PATH = "artifacts/saved_data/curated_datasets/wmt24_de_en_curated_5m"
REPO_ID = "neelpawar/wmt24-de-en"
CONFIG_NAME = "full"


def main():
    print(f"Loading full dataset from {FULL_PATH}...", flush=True)
    full = load_from_disk(FULL_PATH)
    print(f"  train: {len(full['train']):,}", flush=True)
    print(f"  validation (old): {len(full['validation']):,}", flush=True)
    print(f"  test (old/possibly bad): {len(full['test']):,}", flush=True)

    print(f"Loading corrected val/test from {GOOD_PATH}...", flush=True)
    good = load_from_disk(GOOD_PATH)
    print(f"  validation (good): {len(good['validation']):,}", flush=True)
    print(f"  test (good): {len(good['test']):,}", flush=True)

    ds = DatasetDict(
        {
            "train": full["train"],
            "validation": good["validation"],
            "test": good["test"],
        }
    )
    print("Final splits:", flush=True)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}", flush=True)

    print(f"Pushing to {REPO_ID} config={CONFIG_NAME} (this will take a while)...", flush=True)
    ds.push_to_hub(
        REPO_ID,
        config_name=CONFIG_NAME,
        commit_message="Add full config (~295M train) with corrected val/test",
        max_shard_size="500MB",
    )
    print("Done.", flush=True)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")


if __name__ == "__main__":
    main()
