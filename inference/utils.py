import torch
import random
import numpy as np
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

def greedy_decode(model, batch, tokenizer_tgt):
    """
    Decodes the output from the model output probabilities using a basic
    argmax function across the output.
    """
    batch_size, max_padding = batch.src.shape
    try:
        bos_id, eos_id, pad_id = tokenizer_tgt.cls_token_id, tokenizer_tgt.sep_token_id, tokenizer_tgt.pad_token_id
    except:
        bos_id, eos_id, pad_id = tokenizer_tgt.bos_token_id, tokenizer_tgt.eos_token_id, tokenizer_tgt.pad_token_id
    bos_tensor = torch.full((batch_size, 1), bos_id)
    predictions_with_bos = bos_tensor
    pbar = tqdm(range(max_padding - 1), leave=False, desc="Decoding")
    for _ in pbar:
        decoder_attn_mask = batch.make_decoder_attn_mask(predictions_with_bos, pad_id)
        output_logprobabilities = model(batch.src, 
                                        predictions_with_bos.to(batch.src.device), 
                                        decoder_attn_mask.to(batch.src.device))
        predictions = torch.argmax(output_logprobabilities, dim=2).detach().cpu()
        predictions_with_bos = torch.cat([bos_tensor, predictions], dim=1)
    return predictions_with_bos

def probabilistic_decode(model, batch, tokenizer_tgt):
    """
    """
    # TODO: remove extra code from decode functions and consolidate into one predict function. 
    #       code in decode should only be argmax, or probmax
    batch_size, max_padding = batch.src.shape
    bos_id, eos_id, pad_id = tokenizer_tgt.vocab(["<s>", "</s>", "[PAD]"])
    bos_tensor = torch.full((batch_size, 1), bos_id)
    predictions_with_bos = bos_tensor
    for i in range(max_padding - 1):
        decoder_attn_mask = batch.make_decoder_attn_mask(predictions_with_bos, pad_id)
        output_logprobabilities = model(batch.src, predictions_with_bos, decoder_attn_mask)
        # probabilistic selection
        predictions = _prob_argmax(output_logprobabilities)
        predictions_with_bos = torch.cat([bos_tensor, predictions], dim=1).to(batch.src.device)
    return predictions_with_bos.detach().cpu()

def _prob_argmax(logprobs_tensor):
    '''
    Receives a 1D array of log probabilities and makes a probabilistic selection. 
    Current selections are quite noisy, although the implementation has been proven
    to be correct.
    '''
    logprobs_array = logprobs_tensor.detach().numpy()
    batch_size, seq_len, vocab_size = logprobs_array.shape
    result = torch.zeros((batch_size, seq_len), dtype=int)
    for b in range(batch_size):
        for s in range(seq_len):
            logprobs = logprobs_array[b,s,:]
            probs = np.exp(logprobs)
            indices = np.arange(len(probs))
            selected_index = np.random.choice(indices, p=probs)
            result[b,s] = selected_index
    return result

class BleuUtils:        
    # TODO: better name for this class's methods or consolidate into 1/2 methods or move into translator class 
    # which does seem to have similar already
    @staticmethod
    def evaluate_bleu_random_batches(model, dataloader, tokenizer_tgt, num_batches=5, seed=42):
        """
        Evaluates the BLEU score of the model on a random sample of batches from the dataloader.
        args:
            model: the model to evaluate
            dataloader: the dataloader to evaluate on
            tokenizer_tgt: the tokenizer for the target language
            num_batches: the number of batches to evaluate on
            seed: the seed for the random number generator
        returns:
            average_bleu: the average BLEU score over the sampled batches
        """
        model.eval()
        random.seed(seed)
        with torch.no_grad():
            # Convert dataloader to a list of batches
            all_batches = list(dataloader)
            # Randomly sample num_batches from the list
            random_batches = random.sample(all_batches, min(num_batches, len(all_batches)))
            bleu_scores = []
            pbar = tqdm(random_batches, leave=False, desc=f"Evaluating BLEU on {num_batches} random batches")
            for batch in pbar:
                bleu = BleuUtils.evaluate_bleu_one_batch(model, batch, tokenizer_tgt)
                bleu_scores.append(bleu)
                del batch
            average_bleu = sum(bleu_scores) / len(bleu_scores)
            return average_bleu

    @staticmethod
    def evaluate_bleu_one_batch(model, batch, tokenizer_tgt):
        """
        Evaluates the BLEU score of the model on one batch.
        """
        model.eval()
        with torch.no_grad():
            predictions = greedy_decode(model, batch, tokenizer_tgt)
            # src_sentences = tokenizer_src.batch_decode(batch.src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            tgt_sentences = tokenizer_tgt.batch_decode(batch.tgt_label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predicted_sentences = tokenizer_tgt.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            bleu = BleuUtils.compute_batch_bleu(predicted_sentences, tgt_sentences)
            return bleu

    @staticmethod
    def compute_batch_bleu(predicted_sentences, actual_sentences):
        # convert tokens to sentences
        # convert to lists of words format required by bleu function
        predicted_sentences_list = [sentence.split() for sentence in predicted_sentences]
        actual_sentences_list = [[sentence.split()] for sentence in actual_sentences]        
        # compute bleu
        bleu = bleu_score(predicted_sentences_list, actual_sentences_list)
        return bleu