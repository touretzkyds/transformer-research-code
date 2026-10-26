import numpy as np
import torch

# Padded tokens per batch: 512 sentences of length 20, or 20 sentences of length 512.
TOKENS_PER_BATCH = 512 * 20
# Attention memory grows with length squared: a few ~1900-token pairs in one batch run out of memory on a 32 GB V100.
MAX_TOKENS_PER_SENTENCE = 512


def token_lengths(dataset, tokenizer_src, tokenizer_tgt, language_pair, chunk_size=100_000):
    """Length of the longer side of every sentence pair, in tokens, after truncation to MAX_TOKENS_PER_SENTENCE."""
    src_lang, tgt_lang = language_pair
    lengths = []
    for start in range(0, len(dataset), chunk_size):
        pairs = dataset[start:start + chunk_size]["translation"]
        src = tokenizer_src([pair[src_lang] for pair in pairs], truncation=True, max_length=MAX_TOKENS_PER_SENTENCE)["input_ids"]
        tgt = tokenizer_tgt([pair[tgt_lang] for pair in pairs], truncation=True, max_length=MAX_TOKENS_PER_SENTENCE)["input_ids"]
        lengths.extend(max(len(s), len(t)) for s, t in zip(src, tgt))
    return lengths


class TokenBudgetBatchSampler:
    """
    Dynamic batching as in "Attention Is All You Need": sentence pairs of similar length are
    grouped so every batch holds about TOKENS_PER_BATCH padded tokens. Short sentences make
    large batches and long sentences make small ones. Batches stay fixed; their order is
    reshuffled every epoch.
    """
    def __init__(self, lengths, seed, tokens_per_batch=TOKENS_PER_BATCH):
        self.batches = []
        batch, longest = [], 0
        for i in np.argsort(lengths, kind="stable").tolist():
            longest = max(longest, lengths[i])
            if batch and (len(batch) + 1) * longest > tokens_per_batch:
                self.batches.append(batch)
                batch, longest = [], lengths[i]
            batch.append(i)
        self.batches.append(batch)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        print(f"Dynamic batching: {len(lengths)} pairs in {len(self.batches)} batches, longest pair {max(lengths)} tokens")

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for b in torch.randperm(len(self.batches), generator=self.generator).tolist():
            yield self.batches[b]
