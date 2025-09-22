#!/bin/bash
# Example usage script for create_top5_dataset.py

# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Example 1: Create dataset from top 5 performing subsets
python create_top5_dataset.py \
    --results-path artifacts/wmt24_samples_trial_500/translation_results.json \
    --dataset-path artifacts/saved_data/tokenized_data/wmt24_de_en \
    --repo-name "your-username/wmt24-top5-de-en" \
    --top-n 5 \
    --save-locally \
    --output-dir artifacts/top5_dataset

# Example 2: Create dataset from top 3 performing subsets (private repo)
python create_top5_dataset.py \
    --results-path artifacts/wmt24_samples_trial_500/translation_results.json \
    --dataset-path artifacts/saved_data/tokenized_data/wmt24_de_en \
    --repo-name "your-username/wmt24-top3-de-en" \
    --top-n 3 \
    --private \
    --save-locally

# Example 3: Use custom stats path
python create_top5_dataset.py \
    --results-path artifacts/wmt24_samples_trial_500/translation_results.json \
    --dataset-path artifacts/saved_data/tokenized_data/wmt24_de_en \
    --stats-path /path/to/custom/train.stats.json \
    --repo-name "your-username/wmt24-top5-custom" \
    --top-n 5


