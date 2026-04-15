import json
import os
import sys
from pathlib import Path
import time
import torch.nn as nn
import math
# Repo root (parent of final_project/) so `gpt` and other top-level packages resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from gpt.gpt_model import GPTModel, load_weights_into_gpt
from gpt.gpt_download import download_and_load_gpt2


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        return nn.functional.linear(x, combined_weight, self.linear.bias)

class SupportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False, balance_dataset=True):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        if balance_dataset:
            self.data = self._create_balanced_dataset()

        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data['text']
        ]
        if not no_padding:
            # Pad sequences to longest sequence
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]
        
        unique_labels = sorted(self.data['label'].unique())
        self.label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
        self.id_to_label = unique_labels

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label_str = self.data.iloc[idx]['label']
        label_id = self.label_to_id[label_str]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

    @property
    def num_classes(self):
        return len(self.label_to_id)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        return max(len(tokenizer.encode(text)) for text in self.data['text'])

    def _create_balanced_dataset(self) -> pd.DataFrame:
        # Count the number of examples per label
        label_counts = self.data['label'].value_counts()
        min_count = label_counts.min()

        # Sample min_count examples for each label
        balanced_df = pd.DataFrame()
        for label, count in label_counts.items():
            label_df = self.data[self.data['label'] ==
                                 label].sample(min_count, random_state=42)
            balanced_df = pd.concat([balanced_df, label_df])

        return balanced_df


def sample_from_dataset(dataset, k, seed=None):
    """Randomly draw up to k examples; each item is (input_ids, label) as torch.long tensors."""
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    n = len(dataset)
    k = min(k, n)
    perm = torch.randperm(n, generator=g)[:k]
    return [dataset[int(i)] for i in perm]

def peek_dataloader_batch(data_loader):
    """First batch from the loader: (inputs [B, T], labels [B]) with default collate_fn."""
    return next(iter(data_loader))

def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # Find the last non-padding token for each sequence in the batch
                pad_token_id = 50256  # <|endoftext|> token used for padding
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

                logits = model(input_batch)  # Logits of last output token
                # Select the logits corresponding to the last real token of each sequence
                batch_size = logits.size(0)
                selected_logits = logits[torch.arange(batch_size), last_token_pos]
                predicted_labels = torch.argmax(selected_logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

    else:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                model_output = model(input_batch)
                if average_embeddings:
                    # Average over the sequence dimension (dim=1)
                    logits = model_output.mean(dim=1)
                else:
                    # Select embeddings at the specified token position
                    logits = model_output[:, trainable_token_pos, :]

                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    return correct_predictions / num_examples

def instantiate_model(choose_model, load_weights: bool):

    BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

    
    
    if not load_weights:
        torch.manual_seed(123)
        model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    if load_weights:
        model, model_config = download_and_load_gpt2(choose_model, base_config=BASE_CONFIG)

    model.eval()
    return model

def calc_loss_batch(input_batch, target_batch, model, device, trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if trainable_token_pos == "flexible":  # Selects the last tokens before the padding tokens
        # From https://github.com/rasbt/LLMs-from-scratch/discussions/434
        # Find the last non-padding token for each sequence in the batch
        pad_token_id = 50256  # <|endoftext|> token used for padding
        mask = input_batch != pad_token_id
        last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

        # Get model outputs
        logits = model(input_batch)  # shape: [batch_size, seq_len, num_classes]

        # Select the logits corresponding to the last real token of each sequence
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]

        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss

    else:
        model_output = model(input_batch)
        if average_embeddings:
            # Average over the sequence dimension (dim=1)
            logits = model_output.mean(dim=1)
        else:
            # Select embeddings at the specified token position
            logits = model_output[:, trainable_token_pos, :]

        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss

def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    loss_eval_steps, loss_eval_examples = [], []
    epoch_end_examples = []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # Use gradient accumulation if accumulation_steps > 1
            # See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html
            # for an explanation
            loss /= accumulation_steps

            loss.backward()  # Calculate loss gradients

            # Use gradient accumulation if accumulation_steps > 1
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()  # Update model weights using loss gradients
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                loss_eval_steps.append(global_step)
                loss_eval_examples.append(examples_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        epoch_end_examples.append(examples_seen)

        if max_steps is not None and global_step > max_steps:
            break

    return (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        loss_eval_steps,
        loss_eval_examples,
        epoch_end_examples,
    )

def replace_linear_with_lora(model: GPTModel, rank, alpha, alternative=False):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss", outfile=None, xlabel="Epochs", xlabel_top="Examples seen"):
    fig, ax1 = plt.subplots(figsize=(5, 3), layout="constrained")

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.ticklabel_format(style="plain", axis="x", useOffset=False)
    if examples_seen is not None and len(examples_seen) == len(train_values):
        ex0, ex1 = int(examples_seen[0]), int(examples_seen[-1])
        ax1.text(
            0.02,
            0.02,
            f"{xlabel_top}: {ex0} … {ex1}",
            transform=ax1.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            color="0.35",
        )
    path = outfile if outfile is not None else f"{label}-plot.pdf"
    plt.savefig(path)
    plt.close(fig)
    # plt.show()

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they are too long
    max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    input_ids = input_ids[:max_len]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # Logits of last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    return predicted_label


def collect_predictions(data_loader, model, device, trainable_token_pos=-1, average_embeddings=False):
    """Run inference on an entire loader, returning (all_true, all_pred) as plain lists."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            if trainable_token_pos == "flexible":
                pad_token_id = 50256
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1
                logits = model(input_batch)
                batch_size = logits.size(0)
                selected_logits = logits[torch.arange(batch_size, device=device), last_token_pos]
            elif average_embeddings:
                selected_logits = model(input_batch).mean(dim=1)
            else:
                selected_logits = model(input_batch)[:, trainable_token_pos, :]

            preds = torch.argmax(selected_logits, dim=-1)
            all_true.extend(target_batch.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    return all_true, all_pred


def detailed_evaluation(test_loader, model, device, dataset, tokenizer, *,
                        trainable_token_pos=-1, average_embeddings=False,
                        max_misclassified=10):
    """Return a dict with classification_report data, confusion matrix, and misclassified examples."""
    y_true, y_pred = collect_predictions(
        test_loader, model, device,
        trainable_token_pos=trainable_token_pos,
        average_embeddings=average_embeddings,
    )
    label_names = list(dataset.id_to_label)
    num_classes = len(label_names)

    # Per-class precision / recall / F1 (manual computation, no sklearn needed at runtime)
    per_class = {}
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for t in y_true if t == c)
        per_class[label_names[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

    # Confusion matrix as nested list [true][pred]
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Misclassified examples
    misclassified = []
    raw_texts = list(dataset.data["text"])
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p and len(misclassified) < max_misclassified:
            misclassified.append({
                "text": raw_texts[idx],
                "true_label": label_names[t],
                "predicted_label": label_names[p],
            })

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "test_accuracy": round(accuracy, 6),
        "classification_report": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": label_names,
        "misclassified_examples": misclassified,
        "y_true": y_true,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Fine-tune a GPT-2 model on support ticket text data'
    )
    parser.add_argument(
        '--model-size',
        type=str, default='gpt2-medium (355M)',
        choices=('gpt2-small (124M)', 'gpt2-medium (355M)', 'gpt2-large (774M)', 'gpt2-xl (1558M)'),
        help='Model size to use for finetuning.'
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        choices=('pretrained', 'random'),
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        '--trainable-layers',
        type=str, default='last_block',
        choices=('all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'),
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'."
        )
    )
    parser.add_argument(
        "--trainable-token-pos",
        type=str,
        default="last",
        choices=('first', 'last', 'flexible'),
        help=(
            "Which token position to train. Options: 'first', 'last', 'flexible'."
        )
    )
    parser.add_argument(
        "--average-embeddings",
        action="store_true",
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context-length",
        type=str,
        default="longest_training_example",
        help=(
            "The context length of the data inputs."
            " Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help=(
            "The LoRA rank when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
        help=(
            "The LoRA alpha value when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--no-padding",
        action="store_true",
        default=False,
        help=(
            "Disable padding, which means each example may have a different length."
            " This requires setting `--batch_size 1`."
        )
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "The batch size used for training."
        )
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help=(
            "Accumulation steps to allow for gradient accumulation."
            " See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html for explanation."
            " For example, setting `batch_size=8` and `accumulation_steps=1` compute the exact same"
            " loss and weight updates as setting `batch_size=1` and `accumulation_steps=8`, however,"
            " the latter setting uses more iterations."
        )
    )
    parser.add_argument(
        "--disable-causal-mask",
        action="store_true",
        default=False,
        help=(
            "Disables the causal attention mask."
        )
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
        help=(
            "Sets the `ignore_index` in the cross-entropy loss."
        )
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=".",
        help="Directory for loss-plot.pdf and accuracy-plot.pdf (created if missing).",
    )
    parser.add_argument(
        "--emit-metrics-json",
        action="store_true",
        help=(
            "Print one line FINETUNE_METRICS_JSON:<json> with curves and final metrics "
            "(for driver scripts such as compare_finetuning.py)."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not write loss/accuracy PDFs (e.g. when a driver script plots).",
    )
    args = parser.parse_args()

    if args.trainable_token_pos == 'first':
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == 'last':
        args.trainable_token_pos = -1
    elif args.trainable_token_pos == 'flexible':
        args.trainable_token_pos = 'flexible'
    else:
        raise ValueError('Invalid --trainable-token-pos argument.')

    ###############################
    # Load model
    ###############################

    if args.weights == 'pretrained':
        load_weights = True
    elif args.weights == 'random':
        load_weights = False
    else:
        raise ValueError('Invalid --weights argument.')

    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False

    ########################################
    # Load dataset now because of circular dependency
    ########################################

    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = None

    if args.no_padding:
        max_length = None
    else:
        if args.context_length == 'model_context_length':
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == 'longest_training_example':
            train_dataset = SupportDataset(csv_file='data/support_tickets_train.csv', tokenizer=tokenizer, max_length=None, no_padding=args.no_padding, balance_dataset=False)
            max_length = train_dataset.max_length
        else:
            try:
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError('Invalid --context-length argument.')

    if train_dataset is None:
        train_dataset = SupportDataset(csv_file='data/support_tickets_train.csv', tokenizer=tokenizer, max_length=max_length, no_padding=args.no_padding, balance_dataset=False)
    val_dataset = SupportDataset(csv_file='data/support_tickets_dev.csv', tokenizer=tokenizer, max_length=train_dataset.max_length, no_padding=args.no_padding, balance_dataset=False)
    test_dataset = SupportDataset(csv_file='data/support_tickets_test.csv', tokenizer=tokenizer, max_length=train_dataset.max_length, no_padding=args.no_padding, balance_dataset=False)

    # print(f'Train dataset: {len(train_dataset)} examples')
    # print(f'Validation dataset: {len(val_dataset)} examples')
    # print(f'Test dataset: {len(test_dataset)} examples')

    ###############################
    # Load model cont.
    ###############################

    if args.model_size == 'gpt2-small (124M)':
        in_features = 768
    elif args.model_size == 'gpt2-medium (355M)':
        in_features = 1024
    elif args.model_size == 'gpt2-large (774M)':
        in_features = 1280
    elif args.model_size == 'gpt2-xl (1558M)':
        in_features = 1600
    else:
        raise ValueError('Invalid --model-size argument.')

    torch.manual_seed(123)
    model.out_head = nn.Linear(in_features=in_features, out_features=train_dataset.num_classes)

    if args.trainable_layers == 'last_layer':
        pass
    elif args.trainable_layers == 'last_block' or args.trainable_layers == 'last_two_blocks':
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        if args.trainable_layers == 'last_two_blocks':
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True
    elif args.trainable_layers == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_layers in ('lora', 'lora_alternative'):
        if args.trainable_layers == 'lora_alternative':
            alternative = True
        else:
            alternative = False
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, alternative=alternative)
    else:
        raise ValueError('Invalid --trainable-layers argument.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ###############################
    # Instantiate dataloaders
    ###############################

    num_workers = 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model.pos_emb.weight.shape[0]}. Reinitialize data sets with "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        loss_eval_steps,
        loss_eval_examples,
        epoch_end_examples,
    ) = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f'Training completed in {execution_time_minutes:.2f} minutes.')

    ###############################
    # Evaluate model
    ###############################

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    ###############################
    # Detailed test-set evaluation
    ###############################
    eval_detail = detailed_evaluation(
        test_loader, model, device, test_dataset, tokenizer,
        trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings,
        max_misclassified=10,
    )

    print("\n=== Classification Report (Test Set) ===")
    hdr = f"{'Class':>25s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Support':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for cls_name, m in eval_detail["classification_report"].items():
        print(f"{cls_name:>25s}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1-score']:6.4f}  {m['support']:7d}")

    print("\n=== Misclassified Examples (Test Set) ===")
    for i, ex in enumerate(eval_detail["misclassified_examples"], 1):
        print(f"  [{i}] TRUE={ex['true_label']:<22s} PRED={ex['predicted_label']:<22s} TEXT={ex['text']}")

    ########################################
    # Plot results
    ########################################
    plot_dir = Path(args.plot_dir)
    if not args.skip_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_values(
            loss_eval_steps,
            loss_eval_examples,
            train_losses,
            val_losses,
            label="loss",
            outfile=str(plot_dir / "loss-plot.pdf"),
            xlabel="Training step",
        )
        plot_values(
            list(range(1, len(train_accs) + 1)),
            epoch_end_examples,
            train_accs,
            val_accs,
            label="accuracy",
            outfile=str(plot_dir / "accuracy-plot.pdf"),
            xlabel="Epoch",
        )

    if args.emit_metrics_json:
        metrics_payload = {
            "trainable_layers": str(args.trainable_layers),
            "num_epochs": int(args.num_epochs),
            "batch_size": int(args.batch_size),
            "model_size": str(args.model_size),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "loss_eval_steps": [int(x) for x in loss_eval_steps],
            "loss_eval_examples": [int(x) for x in loss_eval_examples],
            "train_accs": [float(x) for x in train_accs],
            "val_accs": [float(x) for x in val_accs],
            "epoch_end_examples": [int(x) for x in epoch_end_examples],
            "final_train_accuracy": float(train_accuracy),
            "final_val_accuracy": float(val_accuracy),
            "final_test_accuracy": float(test_accuracy),
            "time_minutes": float(execution_time_minutes),
            "classification_report": eval_detail["classification_report"],
            "confusion_matrix": eval_detail["confusion_matrix"],
            "confusion_matrix_labels": eval_detail["confusion_matrix_labels"],
            "misclassified_examples": eval_detail["misclassified_examples"],
        }
        if str(args.trainable_layers).startswith("lora"):
            metrics_payload["lora_rank"] = int(args.lora_rank)
            metrics_payload["lora_alpha"] = int(args.lora_alpha)
        print("FINETUNE_METRICS_JSON:" + json.dumps(metrics_payload), flush=True)