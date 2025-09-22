import os
import json
import argparse
import random
from typing import Dict, Tuple, List

from datasets import load_from_disk


def load_training_data_stats(stats_path: str) -> dict:
    with open(stats_path, 'r') as f:
        return json.load(f)


def create_name_to_span_mapping(stats: dict) -> Dict[str, Tuple[int, int]]:
    """
    Build mapping from dataset subpart name to its [start_idx, end_idx) span
    within the concatenated WMT24 training set, using the order and counts in
    stats['counts']['selected'].
    """
    selected_counts = stats['counts']['selected']
    name_to_span: Dict[str, Tuple[int, int]] = {}
    current_idx = 0
    for dataset_name, count in selected_counts.items():
        start_idx = current_idx
        end_idx = current_idx + int(count)
        name_to_span[dataset_name] = (start_idx, end_idx)
        current_idx = end_idx
    return name_to_span

essential_keys = ['train.de', 'train.en']

def validate_dataset_keys(dataset_dict):
    for key in essential_keys:
        if key not in dataset_dict:
            raise KeyError(f"Expected key '{key}' in loaded dataset. Available keys: {list(dataset_dict.keys())}")


def sample_indices_for_span(start_idx: int, end_idx: int, num_samples: int, seed: int, random_sample: bool) -> List[int]:
    span_length = max(0, end_idx - start_idx)
    if span_length == 0:
        return []
    k = min(num_samples, span_length)
    if random_sample:
        rng = random.Random(seed)
        selected_relative = rng.sample(range(span_length), k)
    else:
        selected_relative = list(range(k))
    return [start_idx + r for r in selected_relative]


def sanitize_filename(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def calculate_avg_tokens_per_sentence(texts: List[str]) -> float:
    """Calculate average number of tokens per sentence (whitespace-separated)."""
    if not texts:
        return 0.0
    total_tokens = sum(len(text.split()) for text in texts)
    return total_tokens / len(texts)


def save_pairs_as_txt(output_path: str, de_texts: List[str], en_texts: List[str], delimiter: str = ' | '):
    assert len(de_texts) == len(en_texts)
    with open(output_path, 'w') as f:
        for de, en in zip(de_texts, en_texts):
            f.write(f"{de}{delimiter}{en}\n")


def main():
    parser = argparse.ArgumentParser(description="Sample N sentence pairs per WMT24 subpart and save as txt files.")
    parser.add_argument('--dataset-path', type=str, default='artifacts/saved_data/tokenized_data/wmt24_de_en', help='Path to load_from_disk directory for WMT24 dataset.')
    parser.add_argument('--stats-path', type=str, default=None, help='Path to train.stats.json. If not provided, uses $MTDATA/wmt24-eng-deu/train.stats.json')
    parser.add_argument('-n', '--num-samples', type=int, default=10, help='Number of sentence pairs to sample per subpart.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--sequential', action='store_true', help='If set, take the first N pairs instead of random sampling.')
    parser.add_argument('--out-dir', type=str, default='artifacts/wmt24_samples', help='Directory to write per-subpart txt files.')
    parser.add_argument('--prefix', type=str, default='', help='Optional filename prefix for outputs.')
    parser.add_argument('--delimiter', type=str, default=' | ', help='Delimiter between src and tgt texts in output lines.')
    parser.add_argument('--sample-info', type=str, default=None, help='Path to save sampling info JSON. Defaults to {out-dir}/sample_info.json')

    args = parser.parse_args()

    stats_path = args.stats_path
    if stats_path is None:
        mtd = os.getenv('MTDATA')
        if not mtd:
            raise EnvironmentError("MTDATA env var not set and --stats-path not provided. Provide --stats-path or export MTDATA.")
        stats_path = os.path.join(mtd, 'wmt24-eng-deu', 'train.stats.json')

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at: {stats_path}")

    # Load artifacts
    dataset_dict = load_from_disk(args.dataset_path)
    validate_dataset_keys(dataset_dict)

    stats = load_training_data_stats(stats_path)
    name_to_span = create_name_to_span_mapping(stats)

    os.makedirs(args.out_dir, exist_ok=True)

    sample_info = {}

    # Process each subpart
    for dataset_name, (start_idx, end_idx) in name_to_span.items():
        indices = sample_indices_for_span(start_idx, end_idx, args.num_samples, args.seed, random_sample=not args.sequential)
        if not indices:
            print(f"Skipping {dataset_name}: empty span ({start_idx}, {end_idx})")
            continue

        # Efficiently gather texts using .select on absolute indices
        de_samples = dataset_dict['train.de'].select(indices)['text']
        en_samples = dataset_dict['train.en'].select(indices)['text']

        # Calculate average tokens per sentence
        avg_tokens_de = calculate_avg_tokens_per_sentence(de_samples)
        avg_tokens_en = calculate_avg_tokens_per_sentence(en_samples)

        # Save sampled pairs
        filename = f"{args.prefix}{sanitize_filename(dataset_name)}.txt"
        out_path = os.path.join(args.out_dir, filename)
        save_pairs_as_txt(out_path, de_samples, en_samples, delimiter=args.delimiter)

        # Store sampling info for later use
        sample_info[dataset_name] = {
            "filename": filename,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "sampled_indices": indices,
            "num_samples": len(de_samples),
            "avg_tokens_de": avg_tokens_de,
            "avg_tokens_en": avg_tokens_en
        }

        print(f"Saved {len(de_samples)} pairs for {dataset_name} -> {out_path} (avg tokens: DE={avg_tokens_de:.1f}, EN={avg_tokens_en:.1f})")

    # Save sampling info
    sample_info_path = args.sample_info or os.path.join(args.out_dir, 'sample_info.json')
    with open(sample_info_path, 'w') as f:
        json.dump(sample_info, f, indent=2)
    print(f"Saved sampling info to: {os.path.abspath(sample_info_path)}")

    print(f"Done. Outputs in: {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main() 