import torch
from torchmetrics.functional.text.bleu import bleu_score


def greedy_decode(model, batch, max_padding, tokenizer_tgt):
    """
    Decodes the output from the model output probabilities using a basic
    argmax function across the output.
    """
    # Get device from model (handle DataParallel wrapper)
    actual_model = model.module if hasattr(model, 'module') else model
    device = next(actual_model.parameters()).device
    
    # Move source tokens to device once
    src_tokens = batch.src_tokens.to(device)
    batch_size = src_tokens.shape[0]
    pad_token_id = batch.pad_token_id
    sep_token_id = tokenizer_tgt.sep_token_id if hasattr(tokenizer_tgt, 'sep_token_id') and tokenizer_tgt.sep_token_id is not None else None
    
    # Start with BOS/CLS token
    predictions_with_bos = torch.full((batch_size, 1), tokenizer_tgt.cls_token_id, device=device, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        # Encode source tokens once (reuse across iterations)
        encoder_output = actual_model.encode(src_tokens)
        
        # pbar = tqdm(range(max_padding - 1), leave=False, desc="Decoding")
        for i in range(max_padding - 1):
            # Create decoder attention mask for current predictions (don't use prepare_for_inference which modifies batch)
            batch.create_decoder_attention_mask(predictions_with_bos, pad_token_id)
            decoder_attn_mask = batch.decoder_attention_mask.to(device)
            
            # Forward pass - decode using pre-computed encoder output
            decoder_output = actual_model.decode(predictions_with_bos, encoder_output, decoder_attn_mask)
            output_logprobabilities = actual_model.linear_and_softmax_layers(decoder_output)
            
            # Verify output shape: should be (batch_size, seq_len, vocab_size)
            # where seq_len == predictions_with_bos.shape[1]
            assert output_logprobabilities.shape[:2] == predictions_with_bos.shape[:2], \
                f"Output shape {output_logprobabilities.shape[:2]} doesn't match input shape {predictions_with_bos.shape[:2]}"
            
            # Only take the last token's prediction (autoregressive decoding)
            last_token_logprobs = output_logprobabilities[:, -1, :]  # (batch_size, vocab_size)
            
            next_token = torch.argmax(last_token_logprobs, dim=1, keepdim=True)  # (batch_size, 1)
            
            # Concatenate new token directly on GPU (more efficient)
            predictions_with_bos = torch.cat([predictions_with_bos, next_token], dim=1)
            
            # Early stopping: stop when all sequences have generated SEP token
            if sep_token_id is not None:
                # Check if all sequences already have at least one SEP token in their predictions
                has_sep = (predictions_with_bos == sep_token_id).any(dim=1)  # (batch_size,)
                if has_sep.all():
                    break
            
            # Clean up intermediate tensors
            del output_logprobabilities, last_token_logprobs, next_token, decoder_attn_mask
        batch.offload()
    
    # Move final predictions to CPU before returning
    result = predictions_with_bos.cpu()
    
    del predictions_with_bos, src_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


class BleuUtils:        
    # TODO: see if we can consolidate into 1/2 methods or move into translator class 
    # which does seem to have similar already.
    def __init__(self, config, logger):
        self.config = config
        self.max_padding_train = config.model.max_padding_train
        self.max_padding_test = config.model.max_padding_test
        self.tokenizer_tgt = config.extras.tokenizer_tgt
        self.tokenizer_src = config.extras.tokenizer_src
        self.device = torch.device(config.hardware.device)
        self.logger = logger
        
    def evaluate_bleu(self, model, dataloader, epoch, split_name):
        model.eval()
        with torch.no_grad():
            bleu_scores = []
            print(f"Evaluating BLEU on {split_name} set with {len(dataloader)} batches")
            for i, batch in enumerate(dataloader):
                bleu = self.evaluate_bleu_one_batch(model, batch, save_translations=(i == 0), epoch=epoch, split_name=split_name) # save translations for first batch only
                bleu_scores.append(bleu)
                # Clean up batch and clear cache
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            return average_bleu

    def evaluate_bleu_one_batch(self, model, batch, save_translations=False, epoch=None, split_name=None):
        """
        Evaluates the BLEU score of the model on one batch.
        """
        model.eval()
        max_padding = self.max_padding_test if split_name == "test" else self.max_padding_train
        with torch.no_grad():
            # Prepare batch: shift targets to get tgt_label for decoding
            batch.shift_tgts()
            predictions = greedy_decode(model, batch, max_padding, self.tokenizer_tgt)
            
            # Remove CLS token from predictions to match tgt_label format
            # predictions start with CLS, but tgt_label doesn't have CLS (it was removed in shift_tgts)
            # So we remove the first token from predictions for fair comparison
            if predictions.shape[1] > 0:
                predictions_no_cls = predictions[:, 1:]  # Remove first token (CLS)
            else:
                predictions_no_cls = predictions
            
            # Decode both predictions and targets
            tgt_sentences = self.tokenizer_tgt.batch_decode(batch.tgt_label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predicted_sentences = self.tokenizer_tgt.batch_decode(predictions_no_cls, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Compute BLEU
            bleu = self.compute_batch_bleu(predicted_sentences, tgt_sentences)
            
            if save_translations:
                src_sentences = self.tokenizer_src.batch_decode(batch.src_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True) # only used for logging
                self.logger.save_translation(src_sentences[:5], tgt_sentences[:5], predicted_sentences[:5], predictions_no_cls[:5], bleu, epoch=epoch, split_name=split_name)
                
            # Clean up
            del predictions, predictions_no_cls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return bleu

    @staticmethod
    def compute_batch_bleu(predicted_sentences, actual_sentences):
        references = [[sentence] for sentence in actual_sentences]
        return float(bleu_score(predicted_sentences, references))