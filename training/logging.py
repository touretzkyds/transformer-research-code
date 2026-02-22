import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.interpolate as interp
from tqdm import tqdm
from numerize.numerize import numerize as nu

class BaseLogger:
    def __init__(self, config):
        self.config = config
        self.title_dict = {
            "N": config.model.N,
            "name": config.dataset.name,
            "size": nu(config.dataset.size),
            "bs": config.training.batch_size,
            "seed": config.training.random_seed,
            "seq": config.model.max_padding_train,
        }

    def create_dirs(self, dirs):
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
            # empty the directory
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))


class TrainingLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = defaultdict(list)
        common_save_path = os.path.join(*[f"{k}_{str(v)}" for k, v in self.title_dict.items()])
        self.loss_save_dir = os.path.join(config.logging.artifacts_dir, "loss_curves", common_save_path, "loss")
        self.bleu_save_dir = os.path.join(config.logging.artifacts_dir, "loss_curves", common_save_path, "bleu")
        self.raw_values_save_dir = os.path.join(config.logging.artifacts_dir, "loss_curves", common_save_path, "raw_values")
        self.model_save_dir = os.path.join(config.logging.artifacts_dir, "saved_models", common_save_path)
        self.translation_save_dir = os.path.join(config.logging.artifacts_dir, "generated_translations", common_save_path)
        self.create_dirs([self.loss_save_dir, self.bleu_save_dir, self.raw_values_save_dir, self.model_save_dir, self.translation_save_dir])

    def log_metric(self, name, value, step):
        '''
        Appends metric to existing log of its history over training.
        '''
        self.metrics[name].append(value)
        with open(os.path.join(self.raw_values_save_dir, f"{name}.txt"), "a") as f:
            f.write(f"{step}: {value}\n")

    def save_model(self, model, epoch):
        torch.save(model, os.path.join(self.model_save_dir, f"epoch_{epoch:02d}.pt"))

    def save_translation(self, src_sentences, tgt_sentences, pred_sentences, pred_tokens, bleu_score, epoch, split_name):
        save_path = os.path.join(self.translation_save_dir, f"{split_name}_translations.txt")
        print(f"Saving translations to {save_path}")
        with open(save_path, "a") as f:
            f.write("="*100 + "\n")
            f.write(f"Epoch: {epoch} \t|\t BLEU: {bleu_score}\n")
            f.write("="*100 + "\n")
            for src, tgt, pred, pred_tok in zip(src_sentences, tgt_sentences, pred_sentences, pred_tokens):
                f.write(f"Source: {src}\nTarget: {tgt}\nPredicted: {pred}\nPredicted Tokens: {pred_tok}\n")
                f.write("-"*100 + "\n\n")

    def saveplot(self, epoch_num, metric_names, title, plot_type, xlabel="Epoch"): # TODO: there seems to be a plotting error coming up on thorin. Commiting code as is, and can investigate when psc is up
        '''
        Plots and saves the metric history for specified list of metrics.
        '''
        # compute plot limits
        if plot_type == 'loss':
            ylim = (0, 9)
        elif plot_type == 'bleu':
            ylim = (0, 1)
        else: 
            raise ValueError(f"Invalid plot_type '{plot_type}'")
        composed_title = self.format_title(title)
        max_length = max(
            [len(history) for _, history in self.metrics.items()]
        )
    
        fig, ax = plt.subplots(dpi=300)
        for name in metric_names:
            label = name.replace('_',' ').capitalize()
            metric_history = self.interpolate(self.metrics[name], max_length)
            ax.plot(range(1, len(metric_history)+1), metric_history, label=label)
        ax.set_ylim(ylim)
        ax.set_xlim(1, len(metric_history))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_type.capitalize())
        ax.set_title(composed_title, y=1.08)

        # plot secondary axis of epochs
        ax2 = ax.twiny()
        ax2.set_xlim(1, epoch_num)
        ax2.set_xlabel("Epoch")
        # ax2.set_xticks(range(1, epoch_num + 1))
        ax.grid(visible=True)
        ax.legend()

        # save plot
        fig.savefig(os.path.join(self.loss_save_dir, f"{title.lower().replace(' ', '_')}_{plot_type}.png"))
        plt.close()

    def interpolate(self, array, target_length):
        mesh = interp.interp1d(np.arange(len(array)), array)
        interpolated_array = mesh(np.linspace(0,len(array)-1,target_length)).tolist()
        return interpolated_array

    def format_title(self, title):
        segments = [title]
        for k, v in self.title_dict.items():
            segments.append(f"{k.replace('_', ' ').capitalize()}: {v}")
        return (" | ").join(segments)

class TranslationLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.src_sentences = []
        self.tgt_sentences = []
        self.pred_sentences = []
        self.bleu_scores = []
        # create directories required for saving artifacts
        common_save_path = os.path.join(*[f"{k}_{str(v)}" for k, v in self.title_dict.items()])
        self.save_dir = os.path.join(config.logging.artifacts_dir, "generated_translations", common_save_path)
        self.create_dirs([self.save_dir])

    def log_sentence_batch(self, src_sentences, tgt_sentences, pred_sentences, bleu_score):
        '''
        Logs sentences like target sentence, predicted sentence and so on.
        '''
        self.src_sentences.append(src_sentences)
        self.tgt_sentences.append(tgt_sentences)
        self.pred_sentences.append(pred_sentences)
        self.bleu_scores.append(bleu_score)

    def save_as_txt(self, base_path, title, title_dict):
        save_path = (base_path + 
                     f"N{title_dict['N']}/dataset_size_{title_dict['dataset_size']}/{title_dict['dataset_name']}_epoch_{title_dict['epoch']:02d}.txt")
        print(f"Saving translations to {save_path}")
        with open(save_path, "w") as f:
            avg_bleu = round(sum(self.bleu_scores) / len(self.bleu_scores), 4)
            f.write(f"Average BLEU score: {avg_bleu}\n")
            pbar = tqdm(zip(self.src_sentences, self.tgt_sentences, self.pred_sentences))
            for i, (batch_src, batch_tgt, batch_pred) in enumerate(pbar):
                f.write("-"*100 + f"\nBatch {i+1}:\n" + "-"*100 + "\n")
                for j, (src, tgt, pred) in enumerate(zip(batch_src, batch_tgt, batch_pred)):
                    f.write(f"Source: {src}\nTarget: {tgt}\nPredicted: {pred}\n")
                    f.write(f"\n")