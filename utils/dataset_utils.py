"""
Utility functions for dataset inspection and debugging.
"""

import torch
from datasets import DatasetDict
from typing import Dict, Any, Optional
import json


def inspect_dataset(dataset_dict: DatasetDict, max_samples: int = 5) -> Dict[str, Any]:
    """
    Inspect a HuggingFace dataset and return useful information.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to inspect
        max_samples: Maximum number of samples to display
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        "splits": list(dataset_dict.keys()),
        "total_splits": len(dataset_dict),
        "split_info": {}
    }
    
    for split_name, dataset in dataset_dict.items():
        split_info = {
            "num_examples": len(dataset),
            "features": list(dataset.features.keys()),
            "column_names": dataset.column_names,
            "sample_data": []
        }
        
        # Get sample data
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            sample_info = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample_info[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sample_values": value.tolist()[:10] if value.numel() <= 100 else "too_large"
                    }
                else:
                    sample_info[key] = str(value)[:100]  # Truncate long strings
            split_info["sample_data"].append(sample_info)
        
        info["split_info"][split_name] = split_info
    
    return info


def print_dataset_info(dataset_dict: DatasetDict, max_samples: int = 3):
    """
    Print formatted dataset information.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to inspect
        max_samples: Maximum number of samples to display
    """
    info = inspect_dataset(dataset_dict, max_samples)
    
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Total splits: {info['total_splits']}")
    print(f"Splits: {', '.join(info['splits'])}")
    print()
    
    for split_name, split_info in info["split_info"].items():
        print(f"Split: {split_name}")
        print(f"  Examples: {split_info['num_examples']:,}")
        print(f"  Features: {split_info['features']}")
        print(f"  Columns: {split_info['column_names']}")
        print()
        
        print("  Sample data:")
        for i, sample in enumerate(split_info["sample_data"]):
            print(f"    Sample {i+1}:")
            for key, value in sample.items():
                if isinstance(value, dict) and "shape" in value:
                    print(f"      {key}: shape={value['shape']}, dtype={value['dtype']}")
                else:
                    print(f"      {key}: {value}")
            print()
        print("-" * 40)


def get_dataset_stats(dataset_dict: DatasetDict) -> Dict[str, Any]:
    """
    Get statistical information about the dataset.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to analyze
        
    Returns:
        Dictionary containing statistical information
    """
    stats = {
        "total_examples": 0,
        "split_stats": {}
    }
    
    for split_name, dataset in dataset_dict.items():
        split_stats = {
            "num_examples": len(dataset),
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0
        }
        
        if len(dataset) > 0:
            # Calculate length statistics for text fields
            lengths = []
            for example in dataset:
                for key, value in example.items():
                    if isinstance(value, str):
                        lengths.append(len(value.split()))
                    elif isinstance(value, torch.Tensor) and value.dim() > 0:
                        lengths.append(value.size(0))
            
            if lengths:
                split_stats["avg_length"] = sum(lengths) / len(lengths)
                split_stats["max_length"] = max(lengths)
                split_stats["min_length"] = min(lengths)
        
        stats["split_stats"][split_name] = split_stats
        stats["total_examples"] += split_stats["num_examples"]
    
    return stats


def save_dataset_info(dataset_dict: DatasetDict, output_path: str):
    """
    Save dataset information to a JSON file.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to analyze
        output_path: Path to save the information
    """
    info = inspect_dataset(dataset_dict)
    stats = get_dataset_stats(dataset_dict)
    
    combined_info = {
        "inspection": info,
        "statistics": stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    print(f"Dataset information saved to {output_path}")


def compare_datasets(dataset1: DatasetDict, dataset2: DatasetDict, name1: str = "Dataset1", name2: str = "Dataset2"):
    """
    Compare two datasets and print differences.
    
    Args:
        dataset1: First dataset to compare
        dataset2: Second dataset to compare
        name1: Name for first dataset
        name2: Name for second dataset
    """
    print("=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)
    
    # Compare splits
    splits1 = set(dataset1.keys())
    splits2 = set(dataset2.keys())
    
    print(f"Splits in {name1}: {splits1}")
    print(f"Splits in {name2}: {splits2}")
    print(f"Common splits: {splits1 & splits2}")
    print(f"Only in {name1}: {splits1 - splits2}")
    print(f"Only in {name2}: {splits2 - splits1}")
    print()
    
    # Compare common splits
    for split in splits1 & splits2:
        len1 = len(dataset1[split])
        len2 = len(dataset2[split])
        print(f"Split '{split}':")
        print(f"  {name1}: {len1:,} examples")
        print(f"  {name2}: {len2:,} examples")
        print(f"  Difference: {len1 - len2:,}")
        print()


def validate_dataset_format(dataset_dict: DatasetDict, expected_features: Optional[list] = None) -> bool:
    """
    Validate that a dataset has the expected format.
    
    Args:
        dataset_dict: Dataset to validate
        expected_features: List of expected feature names
        
    Returns:
        True if dataset format is valid, False otherwise
    """
    if not isinstance(dataset_dict, DatasetDict):
        print("Error: Dataset is not a DatasetDict")
        return False
    
    if len(dataset_dict) == 0:
        print("Error: Dataset is empty")
        return False
    
    # Check if all splits have the same features
    features = None
    for split_name, dataset in dataset_dict.items():
        if features is None:
            features = set(dataset.features.keys())
        elif set(dataset.features.keys()) != features:
            print(f"Error: Split '{split_name}' has different features than others")
            return False
    
    # Check expected features if provided
    if expected_features:
        missing_features = set(expected_features) - features
        if missing_features:
            print(f"Error: Missing expected features: {missing_features}")
            return False
    
    print("Dataset format validation passed")
    return True





