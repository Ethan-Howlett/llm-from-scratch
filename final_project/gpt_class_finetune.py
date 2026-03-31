import os
import sys
from pathlib import Path
import time

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


class SupportDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, balance_dataset=True):
        self.data = pd.read_csv(csv_file)
        if balance_dataset:
            self.data = self._create_balanced_dataset()

        unique_labels = sorted(self.data['label'].unique())
        self.label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
        self.id_to_label = unique_labels

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['text']
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Now we need to truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

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

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

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

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # track number of examples seen
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f'Ep {epoch+1} (Step {global_step:06d}): '
                      f'Train loss {train_loss:.3f}, Val loss {val_loss:.3f}')

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(f'Ep {epoch+1}: Train acc {train_accuracy:.3f}, Val acc {val_accuracy:.3f}')

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Fine-tune a GPT-2 model on support ticket text data'
    )

    parser.add_argument(
        '--test_mode',
        default=False,
        action='store_true',
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the final project (recommended).")
    )

    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding('gpt2')

    train_dataset = SupportDataset(
        csv_file='final_project/data/support_tickets_train.csv',
        tokenizer=tokenizer,
        max_length=None,
        balance_dataset=True
    )

    val_dataset = SupportDataset(
        csv_file='final_project/data/support_tickets_dev.csv',
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
        balance_dataset=False
    )

    test_dataset = SupportDataset(
        csv_file='final_project/data/support_tickets_test.csv',
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
        balance_dataset=False
    )

    print(f'Train dataset: {len(train_dataset)} examples')
    print(f'Validation dataset: {len(val_dataset)} examples')
    print(f'Test dataset: {len(test_dataset)} examples')

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    print(f'Train loader: {len(train_loader)} batches')
    print(f'Validation loader: {len(val_loader)} batches')
    print(f'Test loader: {len(test_loader)} batches')

    ########################################
    # Load pretrained model
    ########################################
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
    else:
        CHOOSE_MODEL = "gpt2-medium (355M)"
        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

        model, model_config = download_and_load_gpt2(CHOOSE_MODEL, base_config=BASE_CONFIG)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ########################################
        # Modify and pretrained model
        ########################################
        for param in model.parameters():
            param.requires_grad = False

        torch.manual_seed(123)

        num_classes = train_dataset.num_classes
        model.out_head = torch.nn.Linear(in_features=model_config['emb_dim'], out_features=num_classes)
        model.to(device)

        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True

        for param in model.final_norm.parameters():
            param.requires_grad = True

        ########################################
        # Finetune modified model
        ########################################

        start_time = time.time()
        torch.manual_seed(123)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

        num_epochs = 5

        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f'Training completed in {execution_time_minutes:.2f} minutes.')

        ########################################
        # Plot results
        ########################################

        # loss plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # accuracy plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

        ########################################
        # Test model
        ########################################
        review_1 = ('I need to update my password for security reasons.')
        predicted_label = classify_review(review_1, model, tokenizer, device, max_length=train_dataset.max_length)
        print(f'Review 1: {review_1}')
        print(f'Predicted label: {predicted_label}')
        print(f'Predicted label name: {train_dataset.id_to_label[predicted_label]}')

        review_2 = ('I want to cancel my subscription.')
        predicted_label = classify_review(review_2, model, tokenizer, device, max_length=train_dataset.max_length)
        print(f'Review 2: {review_2}')
        print(f'Predicted label: {predicted_label}')
        print(f'Predicted label name: {train_dataset.id_to_label[predicted_label]}')

        review_3 = ('My package is late.')
        predicted_label = classify_review(review_3, model, tokenizer, device, max_length=train_dataset.max_length)
        print(f'Review 3: {review_3}')
        print(f'Predicted label: {predicted_label}')
        print(f'Predicted label name: {train_dataset.id_to_label[predicted_label]}')

        ########################################
        # Save model
        ########################################
        model_name = CHOOSE_MODEL.split(' ')[0]
        torch.save(model.state_dict(), f'final_project/models/{model_name}-finetuned.pth')
        print(f'Model saved to {f"final_project/models/{model_name}-finetuned.pth"}')
        
        # How to load the model
        # model_state_dict = torch.load(f'final_project/models/{model_name}-finetuned.pth', map_location=device, weights_only=True)
        # model.load_state_dict(model_state_dict)