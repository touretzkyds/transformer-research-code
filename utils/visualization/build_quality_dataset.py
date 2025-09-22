#!/usr/bin/env python3
"""
Dataset Builder for High-Quality Translation Pairs

This script builds a high-quality dataset by:
1. Sampling N=100 sentence pairs at a time from all available subsets
2. Evaluating each batch with BLEU score using a translation model
3. Keeping only batches that meet a quality threshold
4. Continuing until target dataset size is reached or all data is exhausted

Usage:
    python build_quality_dataset.py --target-size 30000 --bleu-threshold 30.0
"""

import os
import json
import argparse
import random
import time
from typing import Dict, List, Tuple, Set
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datasets import load_from_disk


class QualityDatasetBuilder:
    def __init__(self, 
                 dataset_path: str,
                 stats_path: str,
                 translator_type: str = 'hf',
                 hf_model: str = 'Helsinki-NLP/opus-mt-de-en',
                 device: int = -1,
                 batch_size: int = 16,
                 max_length: int = 256,
                 bleu_threshold: float = 30.0,
                 sample_size: int = 100,
                 target_size: int = 30000,
                 seed: int = 42):
        """
        Initialize the quality dataset builder.
        
        Args:
            dataset_path: Path to the WMT24 dataset
            stats_path: Path to train.stats.json
            translator_type: 'hf' or 'google'
            hf_model: HuggingFace model name
            device: Device for HF pipeline (-1 for CPU, >=0 for CUDA)
            batch_size: Batch size for translation
            max_length: Max length for translation
            bleu_threshold: Minimum BLEU score to keep a batch
            sample_size: Number of sentences to sample per batch
            target_size: Target total number of sentences
            seed: Random seed
        """
        self.dataset_path = dataset_path
        self.stats_path = stats_path
        self.translator_type = translator_type
        self.hf_model = hf_model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.bleu_threshold = bleu_threshold
        self.sample_size = sample_size
        self.target_size = target_size
        self.seed = seed
        
        # Initialize random number generator
        self.rng = random.Random(seed)
        
        # Load dataset and stats
        self._load_dataset()
        self._load_stats()
        self._create_name_to_span_mapping()
        
        # Initialize translator
        self._initialize_translator()
        
        # Tracking variables
        self.quality_pairs = []  # List of (de_text, en_text) tuples
        self.quality_metadata = []  # List of metadata dicts
        self.processed_batches = 0
        self.accepted_batches = 0
        self.rejected_batches = 0
        self.exhausted_subsets = set()
        
        # Statistics
        self.bleu_scores = []
        self.subset_stats = {}
        
    def _load_dataset(self):
        """Load the WMT24 dataset."""
        print("Loading WMT24 dataset...")
        self.dataset_dict = load_from_disk(self.dataset_path)
        
        # Validate essential keys
        essential_keys = ['train.de', 'train.en']
        for key in essential_keys:
            if key not in self.dataset_dict:
                raise KeyError(f"Expected key '{key}' in loaded dataset. Available keys: {list(self.dataset_dict.keys())}")
        
        print(f"Dataset loaded successfully. Total sentences: {len(self.dataset_dict['train.de'])}")
    
    def _load_stats(self):
        """Load training data statistics."""
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)
        print(f"Stats loaded from: {self.stats_path}")
    
    def _create_name_to_span_mapping(self):
        """Create mapping from dataset subpart name to its span in the concatenated dataset."""
        selected_counts = self.stats['counts']['selected']
        self.name_to_span = {}
        current_idx = 0
        
        for dataset_name, count in selected_counts.items():
            start_idx = current_idx
            end_idx = current_idx + int(count)
            self.name_to_span[dataset_name] = (start_idx, end_idx)
            current_idx = end_idx
            
        print(f"Created span mapping for {len(self.name_to_span)} subsets")
    
    def _initialize_translator(self):
        """Initialize the translation model."""
        print(f"Initializing {self.translator_type} translator...")
        
        if self.translator_type == 'google':
            try:
                from google.cloud import translate_v2 as translate
                self.translator = translate.Client()
                print("Google Translate client initialized successfully")
            except ImportError:
                raise ImportError("google-cloud-translate not installed. Run: pip install google-cloud-translate")
            except Exception as e:
                raise Exception(f"Failed to initialize Google Translate client: {e}")
        else:  # HuggingFace
            try:
                from transformers import pipeline
                self.translator = pipeline('translation', model=self.hf_model, device=self.device)
                print(f"HuggingFace translator initialized: {self.hf_model}")
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
            except Exception as e:
                raise Exception(f"Failed to initialize HF translator: {e}")
    
    def _translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of German texts to English."""
        if not texts:
            return []
        
        try:
            if self.translator_type == 'google':
                result = self.translator.translate(
                    texts, 
                    target_language='en', 
                    source_language='de', 
                    format_='text'
                )
                
                if isinstance(result, dict):
                    translated = [result['translatedText']]
                else:
                    translated = [r['translatedText'] for r in result]
                
                # Rate limiting
                time.sleep(0.1)
                return translated
            else:  # HuggingFace
                outputs = self.translator(texts, batch_size=self.batch_size, 
                                        truncation=True, max_length=self.max_length)
                return [o['translation_text'] for o in outputs]
        except Exception as e:
            print(f"Translation error: {e}")
            return texts  # Return original as fallback
    
    def _compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """Compute BLEU score between references and hypotheses."""
        assert len(references) == len(hypotheses)
        
        try:
            import sacrebleu
            return sacrebleu.corpus_bleu(hypotheses, [references]).score
        except ImportError:
            try:
                from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                refs_tok = [[r.split()] for r in references]
                hyps_tok = [h.split() for h in hypotheses]
                return float(corpus_bleu(refs_tok, hyps_tok, 
                                       smoothing_function=SmoothingFunction().method3) * 100.0)
            except ImportError:
                print("Warning: Neither sacrebleu nor nltk available. Install with: pip install sacrebleu nltk")
                return 0.0
            except Exception as e:
                print(f"NLTK BLEU error: {e}")
                return 0.0
        except Exception as e:
            print(f"SacreBLEU error: {e}")
            return 0.0
    
    def _sample_batch_from_subset(self, subset_name: str) -> Tuple[List[str], List[str], List[int]]:
        """Sample a batch of sentences from a specific subset."""
        start_idx, end_idx = self.name_to_span[subset_name]
        span_length = max(0, end_idx - start_idx)
        
        if span_length == 0:
            return [], [], []
        
        # Sample indices
        k = min(self.sample_size, span_length)
        selected_relative = self.rng.sample(range(span_length), k)
        selected_indices = [start_idx + r for r in selected_relative]
        
        # Get texts
        de_samples = self.dataset_dict['train.de'].select(selected_indices)['text']
        en_samples = self.dataset_dict['train.en'].select(selected_indices)['text']
        
        return de_samples, en_samples, selected_indices
    
    def _evaluate_batch_quality(self, de_texts: List[str], en_texts: List[str]) -> Tuple[float, List[str]]:
        """Evaluate the quality of a batch by computing BLEU score."""
        # Translate German to English
        translated_texts = self._translate_batch(de_texts)
        
        # Compute BLEU score
        bleu_score = self._compute_bleu(en_texts, translated_texts)
        
        return bleu_score, translated_texts
    
    def _is_subset_exhausted(self, subset_name: str) -> bool:
        """Check if a subset has been exhausted (no more samples available)."""
        return subset_name in self.exhausted_subsets
    
    def _mark_subset_exhausted(self, subset_name: str):
        """Mark a subset as exhausted."""
        self.exhausted_subsets.add(subset_name)
        print(f"  Subset {subset_name} exhausted")
    
    def _get_available_subsets(self) -> List[str]:
        """Get list of subsets that haven't been exhausted."""
        return [name for name in self.name_to_span.keys() 
                if not self._is_subset_exhausted(name)]
    
    def _update_subset_stats(self, subset_name: str, bleu_score: float, accepted: bool):
        """Update statistics for a subset."""
        if subset_name not in self.subset_stats:
            self.subset_stats[subset_name] = {
                'total_batches': 0,
                'accepted_batches': 0,
                'rejected_batches': 0,
                'bleu_scores': [],
                'best_bleu': 0.0,
                'worst_bleu': float('inf')
            }
        
        stats = self.subset_stats[subset_name]
        stats['total_batches'] += 1
        stats['bleu_scores'].append(bleu_score)
        
        if accepted:
            stats['accepted_batches'] += 1
        else:
            stats['rejected_batches'] += 1
        
        stats['best_bleu'] = max(stats['best_bleu'], bleu_score)
        stats['worst_bleu'] = min(stats['worst_bleu'], bleu_score)
    
    def build_dataset(self) -> Dict:
        """Build the quality dataset by sampling and evaluating batches."""
        print(f"\nStarting dataset building...")
        print(f"Target size: {self.target_size:,} sentences")
        print(f"BLEU threshold: {self.bleu_threshold:.1f}")
        print(f"Batch size: {self.sample_size}")
        print(f"Available subsets: {len(self.name_to_span)}")
        
        # Progress tracking
        pbar = tqdm(total=self.target_size, desc="Building quality dataset")
        
        while len(self.quality_pairs) < self.target_size:
            # Check if all subsets are exhausted
            available_subsets = self._get_available_subsets()
            if not available_subsets:
                print(f"\nAll subsets exhausted. Final dataset size: {len(self.quality_pairs):,}")
                break
            
            # Randomly select a subset
            subset_name = self.rng.choice(available_subsets)
            
            # Sample a batch from the subset
            de_texts, en_texts, indices = self._sample_batch_from_subset(subset_name)
            
            if not de_texts:
                self._mark_subset_exhausted(subset_name)
                continue
            
            # Evaluate batch quality
            bleu_score, translated_texts = self._evaluate_batch_quality(de_texts, en_texts)
            
            # Update statistics
            self.processed_batches += 1
            self.bleu_scores.append(bleu_score)
            
            # Check if batch meets quality threshold
            if bleu_score >= self.bleu_threshold:
                # Accept the batch
                self.quality_pairs.extend(list(zip(de_texts, en_texts)))
                
                # Store metadata
                batch_metadata = {
                    'subset_name': subset_name,
                    'indices': indices,
                    'bleu_score': bleu_score,
                    'batch_id': self.processed_batches,
                    'accepted': True
                }
                self.quality_metadata.append(batch_metadata)
                
                self.accepted_batches += 1
                self._update_subset_stats(subset_name, bleu_score, True)
                
                # Update progress
                pbar.update(len(de_texts))
                pbar.set_postfix({
                    'accepted': self.accepted_batches,
                    'rejected': self.rejected_batches,
                    'bleu': f"{bleu_score:.1f}",
                    'subset': subset_name[:20]
                })
                
                print(f"  ✓ Batch {self.processed_batches}: {subset_name} - BLEU {bleu_score:.1f} - ACCEPTED")
            else:
                # Reject the batch
                self.rejected_batches += 1
                self._update_subset_stats(subset_name, bleu_score, False)
                
                pbar.set_postfix({
                    'accepted': self.accepted_batches,
                    'rejected': self.rejected_batches,
                    'bleu': f"{bleu_score:.1f}",
                    'subset': subset_name[:20]
                })
                
                print(f"  ✗ Batch {self.processed_batches}: {subset_name} - BLEU {bleu_score:.1f} - REJECTED")
            
            # Check if subset is exhausted (approximate check)
            start_idx, end_idx = self.name_to_span[subset_name]
            remaining_samples = end_idx - start_idx - (self.processed_batches * self.sample_size)
            if remaining_samples < self.sample_size:
                self._mark_subset_exhausted(subset_name)
        
        pbar.close()
        
        # Prepare results
        results = {
            'dataset_info': {
                'target_size': self.target_size,
                'actual_size': len(self.quality_pairs),
                'bleu_threshold': self.bleu_threshold,
                'sample_size': self.sample_size,
                'seed': self.seed
            },
            'processing_stats': {
                'total_batches_processed': self.processed_batches,
                'accepted_batches': self.accepted_batches,
                'rejected_batches': self.rejected_batches,
                'acceptance_rate': self.accepted_batches / max(1, self.processed_batches),
                'exhausted_subsets': list(self.exhausted_subsets)
            },
            'quality_stats': {
                'mean_bleu': np.mean(self.bleu_scores) if self.bleu_scores else 0.0,
                'median_bleu': np.median(self.bleu_scores) if self.bleu_scores else 0.0,
                'min_bleu': min(self.bleu_scores) if self.bleu_scores else 0.0,
                'max_bleu': max(self.bleu_scores) if self.bleu_scores else 0.0,
                'std_bleu': np.std(self.bleu_scores) if self.bleu_scores else 0.0
            },
            'subset_stats': self.subset_stats,
            'batch_metadata': self.quality_metadata
        }
        
        return results
    
    def save_dataset(self, output_dir: str, prefix: str = "quality_dataset"):
        """Save the quality dataset and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sentence pairs
        pairs_file = os.path.join(output_dir, f"{prefix}_pairs.txt")
        with open(pairs_file, 'w', encoding='utf-8') as f:
            for de_text, en_text in self.quality_pairs:
                f.write(f"{de_text} | {en_text}\n")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'target_size': self.target_size,
                    'actual_size': len(self.quality_pairs),
                    'bleu_threshold': self.bleu_threshold,
                    'sample_size': self.sample_size,
                    'seed': self.seed
                },
                'processing_stats': {
                    'total_batches_processed': self.processed_batches,
                    'accepted_batches': self.accepted_batches,
                    'rejected_batches': self.rejected_batches,
                    'acceptance_rate': self.accepted_batches / max(1, self.processed_batches),
                    'exhausted_subsets': list(self.exhausted_subsets)
                },
                'quality_stats': {
                    'mean_bleu': np.mean(self.bleu_scores) if self.bleu_scores else 0.0,
                    'median_bleu': np.median(self.bleu_scores) if self.bleu_scores else 0.0,
                    'min_bleu': min(self.bleu_scores) if self.bleu_scores else 0.0,
                    'max_bleu': max(self.bleu_scores) if self.bleu_scores else 0.0,
                    'std_bleu': np.std(self.bleu_scores) if self.bleu_scores else 0.0
                },
                'subset_stats': self.subset_stats
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset saved to:")
        print(f"  Pairs: {pairs_file}")
        print(f"  Metadata: {metadata_file}")
        
        return pairs_file, metadata_file


def main():
    parser = argparse.ArgumentParser(description="Build high-quality translation dataset with BLEU filtering.")
    parser.add_argument('--dataset-path', type=str, 
                       default='artifacts/saved_data/tokenized_data/wmt24_de_en',
                       help='Path to WMT24 dataset')
    parser.add_argument('--stats-path', type=str, default=None,
                       help='Path to train.stats.json. If not provided, uses $MTDATA/wmt24-eng-deu/train.stats.json')
    parser.add_argument('--target-size', type=int, default=30000,
                       help='Target number of high-quality sentence pairs')
    parser.add_argument('--bleu-threshold', type=float, default=30.0,
                       help='Minimum BLEU score to accept a batch')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of sentences to sample per batch')
    parser.add_argument('--translator', type=str, choices=['hf', 'google'], default='hf',
                       help='Translation provider: hf (default) or google')
    parser.add_argument('--hf-model', type=str, default='Helsinki-NLP/opus-mt-de-en',
                       help='HuggingFace model name when --translator=hf')
    parser.add_argument('--device', type=int, default=-1,
                       help='Device for HF pipeline: -1 for CPU, >=0 for CUDA')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for translation')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Max length for translation')
    parser.add_argument('--output-dir', type=str, default='artifacts/quality_datasets',
                       help='Output directory for the quality dataset')
    parser.add_argument('--prefix', type=str, default='quality_dataset',
                       help='Prefix for output files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set stats path
    stats_path = args.stats_path
    if stats_path is None:
        mtd = os.getenv('MTDATA')
        if not mtd:
            raise EnvironmentError("MTDATA env var not set and --stats-path not provided. Provide --stats-path or export MTDATA.")
        stats_path = os.path.join(mtd, 'wmt24-eng-deu', 'train.stats.json')
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at: {stats_path}")
    
    # Initialize builder
    builder = QualityDatasetBuilder(
        dataset_path=args.dataset_path,
        stats_path=stats_path,
        translator_type=args.translator,
        hf_model=args.hf_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        bleu_threshold=args.bleu_threshold,
        sample_size=args.sample_size,
        target_size=args.target_size,
        seed=args.seed
    )
    
    # Build dataset
    start_time = time.time()
    results = builder.build_dataset()
    end_time = time.time()
    
    # Save dataset
    pairs_file, metadata_file = builder.save_dataset(args.output_dir, args.prefix)
    
    # Print summary
    print(f"\n{'='*80}")
    print("DATASET BUILDING COMPLETE")
    print(f"{'='*80}")
    print(f"Target size: {args.target_size:,}")
    print(f"Actual size: {len(builder.quality_pairs):,}")
    print(f"BLEU threshold: {args.bleu_threshold:.1f}")
    print(f"Processing time: {end_time - start_time:.1f} seconds")
    print(f"Total batches processed: {builder.processed_batches}")
    print(f"Accepted batches: {builder.accepted_batches}")
    print(f"Rejected batches: {builder.rejected_batches}")
    print(f"Acceptance rate: {builder.accepted_batches / max(1, builder.processed_batches):.1%}")
    
    if builder.bleu_scores:
        print(f"\nBLEU Statistics:")
        print(f"  Mean: {np.mean(builder.bleu_scores):.2f}")
        print(f"  Median: {np.median(builder.bleu_scores):.2f}")
        print(f"  Min: {min(builder.bleu_scores):.2f}")
        print(f"  Max: {max(builder.bleu_scores):.2f}")
        print(f"  Std: {np.std(builder.bleu_scores):.2f}")
    
    print(f"\nOutput files:")
    print(f"  Pairs: {pairs_file}")
    print(f"  Metadata: {metadata_file}")


if __name__ == '__main__':
    main()
