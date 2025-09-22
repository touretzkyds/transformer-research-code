# Quality Dataset Builder

A tool for building high-quality translation datasets by sampling batches of sentences, evaluating them with BLEU scores, and keeping only the high-quality ones.

## Overview

The Quality Dataset Builder addresses the challenge of creating large, high-quality translation datasets by:

1. **Batch Sampling**: Samples N=100 sentences at a time from all available WMT24 subsets
2. **Quality Evaluation**: Uses a translation model to evaluate each batch with BLEU scores
3. **Selective Filtering**: Keeps only batches that meet a quality threshold
4. **Exhaustive Search**: Continues sampling from all subsets until target size is reached or data is exhausted

## Features

- **Configurable Parameters**: Target size, BLEU threshold, batch size, etc.
- **Multiple Translation Models**: Support for HuggingFace and Google Translate
- **Progress Tracking**: Real-time progress bars and statistics
- **Comprehensive Logging**: Detailed metadata and statistics for each batch
- **Subset Management**: Tracks which subsets have been exhausted
- **Quality Statistics**: Mean, median, min, max BLEU scores and more

## Usage

### Basic Usage

```bash
python utils/visualization/build_quality_dataset.py \
    --target-size 30000 \
    --bleu-threshold 30.0 \
    --output-dir artifacts/quality_datasets \
    --prefix my_quality_dataset
```

### Advanced Usage

```bash
python utils/visualization/build_quality_dataset.py \
    --target-size 100000 \
    --bleu-threshold 35.0 \
    --sample-size 100 \
    --translator hf \
    --hf-model Helsinki-NLP/opus-mt-de-en \
    --device 0 \
    --batch-size 32 \
    --max-length 256 \
    --output-dir artifacts/quality_datasets \
    --prefix high_quality_100k \
    --seed 42
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--target-size` | int | 30000 | Target number of high-quality sentence pairs |
| `--bleu-threshold` | float | 30.0 | Minimum BLEU score to accept a batch |
| `--sample-size` | int | 100 | Number of sentences to sample per batch |
| `--translator` | str | 'hf' | Translation provider: 'hf' or 'google' |
| `--hf-model` | str | 'Helsinki-NLP/opus-mt-de-en' | HuggingFace model name |
| `--device` | int | -1 | Device for HF pipeline (-1 for CPU, >=0 for CUDA) |
| `--batch-size` | int | 16 | Batch size for translation |
| `--max-length` | int | 256 | Max length for translation |
| `--output-dir` | str | 'artifacts/quality_datasets' | Output directory |
| `--prefix` | str | 'quality_dataset' | Prefix for output files |
| `--seed` | int | 42 | Random seed |

## Output Files

The script generates two main output files:

### 1. Pairs File (`{prefix}_pairs.txt`)
Contains the high-quality sentence pairs in the format:
```
German sentence | English sentence
German sentence | English sentence
...
```

### 2. Metadata File (`{prefix}_metadata.json`)
Contains comprehensive metadata including:
- Dataset information (target size, actual size, thresholds)
- Processing statistics (batches processed, acceptance rate)
- Quality statistics (mean, median, min, max BLEU scores)
- Subset statistics (per-subset performance)
- Batch metadata (detailed info for each batch)

## Examples

### Example 1: Small Test Dataset
```bash
python utils/visualization/build_quality_dataset.py \
    --target-size 1000 \
    --bleu-threshold 25.0 \
    --sample-size 50 \
    --output-dir artifacts/test_quality \
    --prefix test_1k
```

### Example 2: Large High-Quality Dataset
```bash
python utils/visualization/build_quality_dataset.py \
    --target-size 100000 \
    --bleu-threshold 40.0 \
    --sample-size 100 \
    --output-dir artifacts/quality_datasets \
    --prefix high_quality_100k
```

### Example 3: Very Large Dataset (3M sentences)
```bash
python utils/visualization/build_quality_dataset.py \
    --target-size 3000000 \
    --bleu-threshold 30.0 \
    --sample-size 100 \
    --output-dir artifacts/quality_datasets \
    --prefix huge_3m_bleu30
```

## Running Examples

### Test Script
Run the test script to verify everything works:
```bash
python test_quality_dataset.py
```

### Example Script
Run the example script to see various usage patterns:
```bash
python examples/build_quality_dataset_example.py
```

## Dependencies

The script requires the following Python packages:
- `datasets` - For loading WMT24 dataset
- `transformers` - For HuggingFace translation models
- `sacrebleu` or `nltk` - For BLEU score computation
- `numpy` - For statistical calculations
- `tqdm` - For progress bars

Install with:
```bash
pip install datasets transformers sacrebleu numpy tqdm
```

For Google Translate support:
```bash
pip install google-cloud-translate
```

## Performance Considerations

- **Translation Speed**: HuggingFace models are faster but Google Translate may have higher quality
- **Memory Usage**: Larger batch sizes use more memory but may be faster
- **GPU Usage**: Set `--device 0` to use GPU for faster translation
- **Quality vs Speed**: Higher BLEU thresholds result in fewer accepted batches but higher quality

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--batch-size` or use CPU (`--device -1`)
2. **Slow Performance**: Use GPU (`--device 0`) or reduce `--sample-size`
3. **Low Acceptance Rate**: Lower `--bleu-threshold` or check translation quality
4. **Dataset Not Found**: Ensure WMT24 dataset is properly loaded and `MTDATA` env var is set

### Debug Mode

For debugging, you can modify the script to add more verbose logging or reduce the target size for testing.

## Algorithm Details

1. **Initialization**: Load WMT24 dataset and create subset mappings
2. **Batch Sampling**: Randomly select a subset and sample N sentences
3. **Translation**: Translate German sentences to English using the chosen model
4. **BLEU Evaluation**: Compute BLEU score between translations and references
5. **Quality Filtering**: Accept batch if BLEU >= threshold, otherwise reject
6. **Progress Tracking**: Update statistics and continue until target reached
7. **Subset Exhaustion**: Mark subsets as exhausted when no more samples available
8. **Output Generation**: Save accepted pairs and comprehensive metadata

## Statistics and Monitoring

The script provides detailed statistics including:
- Real-time progress bars
- Acceptance/rejection rates
- BLEU score distributions
- Per-subset performance metrics
- Processing time estimates
- Memory usage monitoring

## Future Enhancements

Potential improvements for future versions:
- Support for other quality metrics (e.g., COMET, BERTScore)
- Parallel processing for faster evaluation
- Adaptive threshold adjustment
- Quality-based subset prioritization
- Integration with other translation models




