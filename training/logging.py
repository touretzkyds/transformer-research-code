import os
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from numerize.numerize import numerize as nu


class BaseLogger:
    def __init__(self, config):
        self.config = config
        self.title_dict = {
            "Layers": config.model.n_layers,
            "Dataset": config.dataset.name,
            "Size": nu(config.dataset.size),
            "BS": config.training.batch_size,
            "Seed": config.training.random_seed,
            "Seq len": config.model.max_padding,
        }
        self.run_path = os.path.join(*[f"{k.replace(' ', '_').lower()}_{v}" for k, v in self.title_dict.items()])

    def create_dirs(self, dirs):
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))

    def format_title(self, title):
        return " | ".join([title] + [f"{k}: {v}" for k, v in self.title_dict.items()])


class TrainingLogger(BaseLogger):
    labels = {"train_loss": "Train Loss", "val_loss": "Val Loss", "train_bleu": "Train BLEU", "val_bleu": "Val BLEU"}

    def __init__(self, config):
        super().__init__(config)
        self.metrics = defaultdict(list)
        self.experiment_save_dir = os.path.join(config.logging.artifacts_dir, "loss_curves", self.run_path)
        self.raw_values_save_dir = os.path.join(self.experiment_save_dir, "raw_values")
        self.model_save_dir = os.path.join(config.logging.artifacts_dir, "saved_models", self.run_path)
        self.translation_save_dir = os.path.join(config.logging.artifacts_dir, "generated_translations", self.run_path)
        os.makedirs(self.experiment_save_dir, exist_ok=True)
        self.create_dirs([self.raw_values_save_dir, self.model_save_dir, self.translation_save_dir])

    def log_metric(self, name, value, step):
        self.metrics[name].append((step, value))
        with open(os.path.join(self.raw_values_save_dir, f"{name}.txt"), "a") as f:
            f.write(f"{step}: {value}\n")

    def save_model(self, model, step):
        state = model.module if hasattr(model, "module") else model
        torch.save({"update_step": step, "model": state.state_dict()}, os.path.join(self.model_save_dir, f"update_{step:06d}.pt"))

    def log_run_time(self, seconds):
        with open(os.path.join(self.experiment_save_dir, "run_time.txt"), "w") as f:
            f.write(f"{seconds:.1f} s ({seconds / 3600:.2f} h)\n")

    def save_translation(self, src_sentences, tgt_sentences, pred_sentences, pred_tokens, bleu_score, step, split_name):
        save_path = os.path.join(self.translation_save_dir, f"{split_name}_translations.txt")
        print(f"Saving translations to {save_path}")
        with open(save_path, "a") as f:
            f.write("="*100 + "\n")
            f.write(f"Update: {step} \t|\t BLEU: {bleu_score:.2f}\n")
            f.write("="*100 + "\n")
            for src, tgt, pred, pred_tok in zip(src_sentences, tgt_sentences, pred_sentences, pred_tokens):
                f.write(f"Source: {src}\nTarget: {tgt}\nPredicted: {pred}\nPredicted Tokens: {pred_tok}\n")
                f.write("-"*100 + "\n\n")

    def saveplot(self, updates_per_epoch, metric_names, title, plot_type, xlabel="Weight Updates"):
        '''
        Plots metrics against weight updates, with epochs on a secondary x-axis.
        Loss starts at update 0; BLEU starts at its first evaluation.
        '''
        ylim = {"loss": (0, 9), "bleu": (0, 100)}[plot_type]
        first_step = 0 if plot_type == "loss" else self.metrics[metric_names[0]][0][0]
        last_step = max(self.metrics[name][-1][0] for name in metric_names)

        fig, ax = plt.subplots(dpi=300)
        for name in metric_names:
            steps, values = zip(*self.metrics[name])
            marker = None if name == "train_loss" else "o"
            ax.plot(steps, values, label=self.labels[name], marker=marker, markersize=4)
        ax.set_ylim(ylim)
        ax.set_xlim(first_step, last_step)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title)
        ax.grid(visible=True)
        ax.legend()

        ax2 = ax.twiny()
        ax2.set_xlim(first_step / updates_per_epoch, last_step / updates_per_epoch)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_xlabel("Epoch")

        fig.suptitle(self.format_title(title))
        fig.subplots_adjust(top=0.82)
        fig.savefig(os.path.join(self.experiment_save_dir, f"{plot_type}.svg"), format="svg", bbox_inches="tight")
        plt.close()


class TranslationLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.src_sentences = []
        self.tgt_sentences = []
        self.pred_sentences = []
        self.bleu_scores = []
        self.save_dir = os.path.join(config.logging.artifacts_dir, "generated_translations", self.run_path)
        self.create_dirs([self.save_dir])

    def log_sentence_batch(self, src_sentences, tgt_sentences, pred_sentences, bleu_score):
        self.src_sentences.append(src_sentences)
        self.tgt_sentences.append(tgt_sentences)
        self.pred_sentences.append(pred_sentences)
        self.bleu_scores.append(bleu_score)

    def save_as_txt(self, base_path, title, title_dict):
        save_path = (base_path +
                     f"n_layers_{title_dict['n_layers']}/dataset_size_{title_dict['dataset_size']}/{title_dict['dataset_name']}_epoch_{title_dict['epoch']:02d}.txt")
        print(f"Saving translations to {save_path}")
        with open(save_path, "w") as f:
            avg_bleu = sum(self.bleu_scores) / len(self.bleu_scores)
            f.write(f"Average BLEU score: {avg_bleu:.2f}\n")
            pbar = tqdm(zip(self.src_sentences, self.tgt_sentences, self.pred_sentences))
            for i, (batch_src, batch_tgt, batch_pred) in enumerate(pbar):
                f.write("-"*100 + f"\nBatch {i+1}:\n" + "-"*100 + "\n")
                for src, tgt, pred in zip(batch_src, batch_tgt, batch_pred):
                    f.write(f"Source: {src}\nTarget: {tgt}\nPredicted: {pred}\n")
                    f.write(f"\n")