#!/usr/bin/env python3
"""
Test script for the quality dataset builder.
This script tests the functionality with a small dataset size first.
"""

import os
import sys
import subprocess
import time

def test_quality_dataset_builder():
    """Test the quality dataset builder with a small dataset size."""
    
    print("Testing Quality Dataset Builder...")
    print("="*50)
    
    # Test parameters
    test_params = {
        'target_size': 1000,  # Small test size
        'bleu_threshold': 25.0,  # Lower threshold for testing
        'sample_size': 50,  # Smaller batch size for testing
        'output_dir': 'artifacts/test_quality_dataset',
        'prefix': 'test_quality'
    }
    
    # Build command
    cmd = [
        'python', 'utils/visualization/build_quality_dataset.py',
        '--target-size', str(test_params['target_size']),
        '--bleu-threshold', str(test_params['bleu_threshold']),
        '--sample-size', str(test_params['sample_size']),
        '--output-dir', test_params['output_dir'],
        '--prefix', test_params['prefix'],
        '--batch-size', '8',  # Smaller batch size for testing
        '--device', '-1'  # Use CPU for testing
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Test parameters: {test_params}")
    print()
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Test completed successfully in {end_time - start_time:.1f} seconds")
            
            # Check if output files were created
            pairs_file = os.path.join(test_params['output_dir'], f"{test_params['prefix']}_pairs.txt")
            metadata_file = os.path.join(test_params['output_dir'], f"{test_params['prefix']}_metadata.json")
            
            if os.path.exists(pairs_file) and os.path.exists(metadata_file):
                print(f"✅ Output files created successfully:")
                print(f"  - {pairs_file}")
                print(f"  - {metadata_file}")
                
                # Check file sizes
                pairs_size = os.path.getsize(pairs_file)
                metadata_size = os.path.getsize(metadata_file)
                print(f"  - Pairs file size: {pairs_size:,} bytes")
                print(f"  - Metadata file size: {metadata_size:,} bytes")
                
                return True
            else:
                print("❌ Output files not found")
                return False
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main test function."""
    print("Quality Dataset Builder Test")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('utils/visualization/build_quality_dataset.py'):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run the test
    success = test_quality_dataset_builder()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()




