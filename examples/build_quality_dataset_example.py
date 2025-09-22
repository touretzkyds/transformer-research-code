#!/usr/bin/env python3
"""
Example usage of the Quality Dataset Builder.

This script demonstrates how to use the build_quality_dataset.py script
to create high-quality translation datasets of various sizes.
"""

import os
import subprocess
import time
from pathlib import Path

def run_dataset_builder(target_size, bleu_threshold, output_name, sample_size=100):
    """Run the dataset builder with specified parameters."""
    
    print(f"\n{'='*60}")
    print(f"Building Quality Dataset: {output_name}")
    print(f"{'='*60}")
    print(f"Target size: {target_size:,} sentences")
    print(f"BLEU threshold: {bleu_threshold:.1f}")
    print(f"Sample size: {sample_size}")
    
    # Build command
    cmd = [
        'python', 'utils/visualization/build_quality_dataset.py',
        '--target-size', str(target_size),
        '--bleu-threshold', str(bleu_threshold),
        '--sample-size', str(sample_size),
        '--output-dir', f'artifacts/quality_datasets/{output_name}',
        '--prefix', output_name,
        '--batch-size', '16',
        '--device', '-1'  # Use CPU
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            end_time = time.time()
            print("✅ SUCCESS!")
            print(f"Completed in {end_time - start_time:.1f} seconds")
            print("\nOutput:")
            print(result.stdout)
            
            if result.stderr:
                print("\nWarnings/Errors:")
                print(result.stderr)
            
            return True
        else:
            print("❌ FAILED!")
            print(f"Return code: {result.returncode}")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT!")
        print("Process timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Main function demonstrating various usage examples."""
    
    print("Quality Dataset Builder - Usage Examples")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('utils/visualization/build_quality_dataset.py'):
        print("❌ Error: Please run this script from the project root directory")
        return
    
    # Create output directory
    os.makedirs('artifacts/quality_datasets', exist_ok=True)
    
    # Example 1: Small test dataset
    print("\nExample 1: Small Test Dataset")
    success1 = run_dataset_builder(
        target_size=1000,
        bleu_threshold=25.0,
        output_name="test_1k_bleu25",
        sample_size=50
    )
    
    if not success1:
        print("❌ Example 1 failed, stopping here")
        return
    
    # Example 2: Medium dataset with higher quality threshold
    print("\nExample 2: Medium Dataset with Higher Quality")
    success2 = run_dataset_builder(
        target_size=10000,
        bleu_threshold=35.0,
        output_name="medium_10k_bleu35",
        sample_size=100
    )
    
    if not success2:
        print("❌ Example 2 failed, stopping here")
        return
    
    # Example 3: Large dataset with very high quality threshold
    print("\nExample 3: Large High-Quality Dataset")
    success3 = run_dataset_builder(
        target_size=50000,
        bleu_threshold=40.0,
        output_name="large_50k_bleu40",
        sample_size=100
    )
    
    if not success3:
        print("❌ Example 3 failed, stopping here")
        return
    
    # Example 4: Very large dataset (this might take a while)
    print("\nExample 4: Very Large Dataset")
    print("⚠️  This might take a long time...")
    
    # Ask user if they want to continue
    response = input("Continue with very large dataset (3M sentences)? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        success4 = run_dataset_builder(
            target_size=3000000,
            bleu_threshold=30.0,
            output_name="huge_3m_bleu30",
            sample_size=100
        )
        
        if success4:
            print("🎉 All examples completed successfully!")
        else:
            print("❌ Example 4 failed")
    else:
        print("Skipping very large dataset example")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("Check the 'artifacts/quality_datasets/' directory for results")
    print("="*60)

if __name__ == '__main__':
    main()




