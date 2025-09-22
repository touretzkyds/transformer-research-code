#!/usr/bin/env python3
"""
Simple test for the quality dataset builder using a mock translator.
This avoids the slow model loading for quick testing.
"""

import os
import sys
import json
import random
import time
from typing import Dict, List, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.visualization.build_quality_dataset import QualityDatasetBuilder


class MockTranslator:
    """Mock translator that returns random BLEU scores for testing."""
    
    def __init__(self, **kwargs):
        self.rng = random.Random(42)
        print("Mock translator initialized (for testing)")
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Return mock translations."""
        # Return the original texts as "translations" for testing
        return texts.copy()


class MockQualityDatasetBuilder(QualityDatasetBuilder):
    """Quality dataset builder with mock translator for testing."""
    
    def _initialize_translator(self):
        """Initialize the mock translator."""
        self.translator = MockTranslator()
    
    def _translate_batch(self, texts: List[str]) -> List[str]:
        """Use mock translator."""
        return self.translator.translate_batch(texts)
    
    def _compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """Return mock BLEU scores for testing."""
        # Return random BLEU scores between 20-60 for testing
        return self.rng.uniform(20.0, 60.0)


def test_mock_dataset_builder():
    """Test the dataset builder with mock translator."""
    
    print("Testing Quality Dataset Builder with Mock Translator")
    print("="*60)
    
    # Test parameters
    test_params = {
        'target_size': 100,
        'bleu_threshold': 40.0,
        'sample_size': 20,
        'output_dir': 'artifacts/test_mock_quality',
        'prefix': 'test_mock'
    }
    
    print(f"Test parameters: {test_params}")
    
    # Set up paths
    dataset_path = 'artifacts/saved_data/tokenized_data/wmt24_de_en'
    stats_path = os.path.join(os.getenv('MTDATA', ''), 'wmt24-eng-deu', 'train.stats.json')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    if not os.path.exists(stats_path):
        print(f"❌ Stats not found at: {stats_path}")
        return False
    
    try:
        # Initialize builder
        builder = MockQualityDatasetBuilder(
            dataset_path=dataset_path,
            stats_path=stats_path,
            bleu_threshold=test_params['bleu_threshold'],
            sample_size=test_params['sample_size'],
            target_size=test_params['target_size'],
            seed=42
        )
        
        print("✅ Builder initialized successfully")
        
        # Build dataset
        start_time = time.time()
        results = builder.build_dataset()
        end_time = time.time()
        
        print(f"✅ Dataset building completed in {end_time - start_time:.1f} seconds")
        
        # Check results
        print(f"Target size: {test_params['target_size']}")
        print(f"Actual size: {len(builder.quality_pairs)}")
        print(f"Processed batches: {builder.processed_batches}")
        print(f"Accepted batches: {builder.accepted_batches}")
        print(f"Rejected batches: {builder.rejected_batches}")
        
        if builder.bleu_scores:
            print(f"BLEU scores - Mean: {sum(builder.bleu_scores)/len(builder.bleu_scores):.2f}")
        
        # Save dataset
        pairs_file, metadata_file = builder.save_dataset(
            test_params['output_dir'], 
            test_params['prefix']
        )
        
        print(f"✅ Dataset saved:")
        print(f"  - {pairs_file}")
        print(f"  - {metadata_file}")
        
        # Verify files exist and have content
        if os.path.exists(pairs_file) and os.path.exists(metadata_file):
            pairs_size = os.path.getsize(pairs_file)
            metadata_size = os.path.getsize(metadata_file)
            print(f"  - Pairs file size: {pairs_size:,} bytes")
            print(f"  - Metadata file size: {metadata_size:,} bytes")
            
            # Check if we have the expected number of lines
            with open(pairs_file, 'r') as f:
                lines = f.readlines()
            print(f"  - Number of pairs: {len(lines)}")
            
            return True
        else:
            print("❌ Output files not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Mock Quality Dataset Builder Test")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('utils/visualization/build_quality_dataset.py'):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run the test
    success = test_mock_dataset_builder()
    
    if success:
        print("\n🎉 Mock test passed!")
        print("\nNote: This test used a mock translator with random BLEU scores.")
        print("For real testing, use the actual script with a real translator.")
        sys.exit(0)
    else:
        print("\n💥 Mock test failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()




