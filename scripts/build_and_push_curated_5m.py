#!/usr/bin/env python3
"""Build curated_5m config from the local 5m tokenized cache and push to Hub."""

import os
import os.path as op
import sys

sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))

from datasets import Dataset, DatasetDict, Features, load_from_disk
from datasets.features.translation import Translation

CACHE_PATH = "artifacts/saved_data/tokenized_data/wmt24_de_en_5m"
OUT_PATH = "artifacts/saved_data/curated_datasets/wmt24_de_en_curated_5m"
REPO_ID = "neelpawar/wmt24-de-en"
CONFIG_NAME = "curated_5m"
LANGUAGE_PAIR = ["de", "en"]


def pair_to_translation(src_ds, tgt_ds, src_lang, tgt_lang):
    n = min(len(src_ds), len(tgt_ds))
    if len(src_ds) != len(tgt_ds):
        print(f"  size mismatch: src={len(src_ds)}, tgt={len(tgt_ds)}; using {n}")
    src_ds = src_ds.select(range(n))
    tgt_ds = tgt_ds.select(range(n))

    print(f"  loading text columns ({n:,} rows)...", flush=True)
    paired = Dataset.from_dict({"de": src_ds["text"], "en": tgt_ds["text"]})

    features = Features({"translation": Translation(languages=[src_lang, tgt_lang])})

    def pack(batch):
        return {
            "translation": [
                {src_lang: d, tgt_lang: e} for d, e in zip(batch["de"], batch["en"])
            ]
        }

    print("  packing into translation feature...", flush=True)
    return paired.map(
        pack,
        batched=True,
        batch_size=10_000,
        remove_columns=["de", "en"],
        features=features,
        num_proc=min(os.cpu_count() or 4, 8),
        desc="pack translation",
    )


def main():
    src_lang, tgt_lang = LANGUAGE_PAIR
    print(f"Loading cache from {CACHE_PATH}...")
    cache = load_from_disk(CACHE_PATH)
    for key in cache:
        print(f"  {key}: {len(cache[key]):,}")

    print("Building translation splits...")
    split_map = {
        "train": ("train.de", "train.en"),
        "validation": ("val.de", "val.en"),
        "test": ("test.de", "test.en"),
    }
    new_splits = {}
    for new_name, (src_key, tgt_key) in split_map.items():
        print(f"  === {new_name} ===", flush=True)
        new_splits[new_name] = pair_to_translation(
            cache[src_key], cache[tgt_key], src_lang, tgt_lang
        )
        print(f"  {new_name}: {len(new_splits[new_name]):,} rows", flush=True)

    ds = DatasetDict(new_splits)
    print(f"Saving to {OUT_PATH}...", flush=True)
    os.makedirs(op.dirname(OUT_PATH), exist_ok=True)
    ds.save_to_disk(OUT_PATH)
    print("Saved.", flush=True)

    print(f"Pushing to {REPO_ID} config={CONFIG_NAME}...", flush=True)
    ds.push_to_hub(
        REPO_ID,
        config_name=CONFIG_NAME,
        commit_message="Add curated_5m config subset",
        max_shard_size="500MB",
    )
    print("Done.", flush=True)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")


if __name__ == "__main__":
    main()
