import os
import json
import argparse
import time
from typing import Dict, List, Tuple
from pathlib import Path
import glob
from tqdm import tqdm


class HFTranslator:
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-de-en', device: int = -1, batch_size: int = 16, max_length: int = 256):
        """Initialize HuggingFace translation pipeline."""
        try:
            from transformers import pipeline
            self.pipe = pipeline('translation', model=model_name, device=device)
            self.batch_size = batch_size
            self.max_length = max_length
            print(f"Loaded HF model: {model_name}")
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        except Exception as e:
            raise Exception(f"Failed to initialize HF translator: {e}")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of German texts to English."""
        if not texts:
            return []
        
        try:
            outputs = self.pipe(texts, batch_size=self.batch_size, truncation=True, max_length=self.max_length)
            return [o['translation_text'] for o in outputs]
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original texts as fallback
            return texts


class GoogleTranslator:
    def __init__(self, batch_size: int = 16, delay: float = 0.1):
        """
        Initialize Google Translate client.
        Requires GOOGLE_APPLICATION_CREDENTIALS env var for service account.
        """
        try:
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
            self.batch_size = batch_size
            self.delay = delay  # Rate limiting delay between batches
        except ImportError:
            raise ImportError("google-cloud-translate not installed. Run: pip install google-cloud-translate")
        except Exception as e:
            raise Exception(f"Failed to initialize Google Translate client: {e}")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of German texts to English."""
        if not texts:
            return []
        
        try:
            # API supports batching
            result = self.client.translate(
                texts, 
                target_language='en', 
                source_language='de', 
                format_='text'
            )
            
            # Result can be dict or list of dicts depending on input
            if isinstance(result, dict):
                translated = [result['translatedText']]
            else:
                translated = [r['translatedText'] for r in result]
            
            # Rate limiting
            time.sleep(self.delay)
            return translated
            
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original texts as fallback
            return texts


def get_translator(translator_type: str, **kwargs):
    """Get translator instance based on type."""
    if translator_type == 'google':
        return GoogleTranslator(**kwargs)
    else:
        return HFTranslator(**kwargs)


def discover_sample_files(samples_dir: str) -> Dict[str, str]:
    """Automatically discover all txt files in the samples directory."""
    sample_files = {}
    
    # Find all txt files
    txt_pattern = os.path.join(samples_dir, "*.txt")
    txt_files = glob.glob(txt_pattern)
    
    for file_path in txt_files:
        # Extract dataset name from filename (remove .txt extension)
        filename = os.path.basename(file_path)
        dataset_name = filename.replace('.txt', '')
        sample_files[dataset_name] = file_path
    
    return sample_files


def load_sampled_file(file_path: str, delimiter: str = ' | ') -> Tuple[List[str], List[str]]:
    """Load German and English texts from a sampled file."""
    de_texts, en_texts = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and delimiter in line:
                parts = line.split(delimiter, 1)
                if len(parts) == 2:
                    de_texts.append(parts[0].strip())
                    en_texts.append(parts[1].strip())
    
    return de_texts, en_texts


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
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
            return float(corpus_bleu(refs_tok, hyps_tok, smoothing_function=SmoothingFunction().method3) * 100.0)
        except ImportError:
            print("Warning: Neither sacrebleu nor nltk available. Install with: pip install sacrebleu nltk")
            return 0.0
        except Exception as e:
            print(f"NLTK BLEU error: {e}")
            return 0.0
    except Exception as e:
        print(f"SacreBLEU error: {e}")
        return 0.0


def save_translation_results(results: Dict, output_path: str):
    """Save translation and BLEU results to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Translate sampled German texts to English and compute BLEU scores.")
    parser.add_argument('--samples-dir', type=str, required=True, help='Directory containing sampled txt files.')
    parser.add_argument('--translator', type=str, choices=['hf', 'google'], default='hf', help='Translation provider: hf (default, offline) or google (requires credentials).')
    parser.add_argument('--hf-model', type=str, default='Helsinki-NLP/opus-mt-de-en', help='HF translation model name when --translator=hf.')
    parser.add_argument('--device', type=int, default=-1, help='Device for HF pipeline: -1 for CPU, >=0 for CUDA device index.')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path for results. Defaults to {samples_dir}/translation_results.json')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for translation.')
    parser.add_argument('--max-length', type=int, default=256, help='Max generated tokens for translation.')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between API batches (seconds, for Google only).')
    parser.add_argument('--delimiter', type=str, default=' | ', help='Delimiter used in sampled files.')
    parser.add_argument('--save-translations', action='store_true', help='Save individual translation files per subpart.')
    parser.add_argument('--translations-dir', type=str, default=None, help='Directory to save translation files. Defaults to {samples_dir}/translations/')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {args.samples_dir}")

    # Discover sample files automatically
    sample_files = discover_sample_files(args.samples_dir)
    if not sample_files:
        raise FileNotFoundError(f"No txt files found in {args.samples_dir}")
    
    print(f"Discovered {len(sample_files)} sample files:")
    for dataset_name, file_path in sample_files.items():
        print(f"{os.path.basename(file_path)}", end=", ")

    # Initialize translator
    try:
        if args.translator == 'google':
            translator = get_translator('google', batch_size=args.batch_size, delay=args.delay)
            print("\nGoogle Translate client initialized successfully")
        else:
            translator = get_translator('hf', model_name=args.hf_model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
            print(f"\nHuggingFace translator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize translator: {e}")
        return

    # Prepare output paths
    output_path = args.output or os.path.join(args.samples_dir, 'translation_results.json')
    translations_dir = args.translations_dir or os.path.join(args.samples_dir, 'translations')
    
    if args.save_translations:
        os.makedirs(translations_dir, exist_ok=True)

    results = {}
    total_translated = 0

    # Process each subpart
    for dataset_name, file_path in sample_files.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Load German and English texts
        de_texts, en_texts = load_sampled_file(file_path, args.delimiter)
        if not de_texts:
            print(f"  Warning: No texts loaded from {file_path}")
            continue

        print(f"  Loaded {len(de_texts)} sentence pairs")

        # Translate German to English in batches
        translated_texts = []
        pbar = tqdm(range(0, len(de_texts), args.batch_size), desc="Translating batch")
        for i in pbar:
            batch = de_texts[i:i + args.batch_size]
            batch_translations = translator.translate_batch(batch)
            translated_texts.extend(batch_translations)
        
        # Compute BLEU score
        bleu_score = compute_bleu(en_texts, translated_texts)
        pbar.set_description(f"{dataset_name}: BLEU {bleu_score:.2f}")
        print(f"  BLEU score: {bleu_score:.2f}")
        # Store results
        results[dataset_name] = {
            "bleu_score": bleu_score,
            "num_samples": len(en_texts),
            "filename": os.path.basename(file_path)
        }

        total_translated += len(en_texts)
        print(f"  BLEU score: {bleu_score:.2f}")

        # Save individual translation file if requested
        if args.save_translations:
            trans_file_path = os.path.join(translations_dir, f"{dataset_name}_translations.txt")
            with open(trans_file_path, 'w', encoding='utf-8') as f:
                for de, en_ref, en_trans in zip(de_texts, en_texts, translated_texts):
                    f.write(f"DE: {de}\n")
                    f.write(f"EN_REF: {en_ref}\n")
                    f.write(f"EN_TRANS: {en_trans}\n")
                    f.write("-" * 80 + "\n")
            print(f"  Saved translations to: {trans_file_path}")

    # Sort results by BLEU score (descending)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['bleu_score'], reverse=True))

    # Save overall results
    save_translation_results(sorted_results, output_path)
    print(f"\nSaved results to: {output_path}")
    print(f"Total sentences translated: {total_translated}")

    # Print ranking
    print("\n" + "="*80)
    print("BLEU SCORE RANKING (Best to Worst)")
    print("="*80)
    for i, (dataset_name, info) in enumerate(sorted_results.items(), 1):
        print(f"{i:2d}. {dataset_name:<40} BLEU: {info['bleu_score']:6.2f} (N={info['num_samples']})")

    # Summary statistics
    bleu_scores = [info['bleu_score'] for info in sorted_results.values()]
    if bleu_scores:
        print(f"\nSummary Statistics:")
        print(f"  Mean BLEU: {sum(bleu_scores)/len(bleu_scores):.2f}")
        print(f"  Median BLEU: {sorted(bleu_scores)[len(bleu_scores)//2]:.2f}")
        print(f"  Min BLEU: {min(bleu_scores):.2f}")
        print(f"  Max BLEU: {max(bleu_scores):.2f}")


if __name__ == '__main__':
    main()
