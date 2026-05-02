"""
Train a 125M GPT-2 on the French BabyLM corpus.

Architecture: 125M GPT-2 (12 layers, d_model=768, 12 heads)
  - Matches fractal-language ablations for direct comparability
  - Causal LM objective

Checkpoint strategy (per BabyLM 2025 eval pipeline):
  - Save at 1M, 2M, 3M, ... 10M words processed
  - Save every 10M words after that (20M, 30M, ... 100M)
  - Each checkpoint is a full HuggingFace model directory

Usage:
  uv run python scripts/train.py
  uv run python scripts/train.py --resume models/chck_50M
  uv run python scripts/train.py --epochs 5 --batch_size 16
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training.

    Note: the submitted checkpoint chck_92M_epoch3 was trained before
    explicit seeding was added, so re-running this script with the
    default seed will not produce a bit-identical checkpoint to the
    one on the BabyLM leaderboard. The canonical model is the one
    released on the HuggingFace Hub; this seed is for reproducibility
    of future runs (ablations, replications, follow-up experiments).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Paths: auto-detect local vs remote
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if os.path.exists("/workspace/babylm"):
    _PROJECT_DIR = "/workspace/babylm"

CORPUS_DIR = os.path.join(_PROJECT_DIR, "corpus", "final")
MODELS_DIR = os.path.join(_PROJECT_DIR, "models")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "tokenizer")

# BabyLM word budget
STRICT_BUDGET = 100_000_000

# Checkpoint schedule: save at these word counts
CHECKPOINT_WORDS = (
    [i * 1_000_000 for i in range(1, 11)]  # 1M..10M
    + [i * 10_000_000 for i in range(2, 11)]  # 20M..100M
)


def parse_args():
    p = argparse.ArgumentParser(description="Train GPT-2 125M on French BabyLM corpus")
    p.add_argument("--corpus", default=os.path.join(CORPUS_DIR, "train_french.txt"),
                    help="Path to training corpus")
    p.add_argument("--epochs", type=int, default=1,
                    help="Number of epochs (each counts toward word budget)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for Python, NumPy, and PyTorch")
    return p.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def train_tokenizer(corpus_path, vocab_size, save_dir):
    """Train a BPE tokenizer on the French corpus."""
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(tokenizer_path)

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    hf_tokenizer.pad_token = "<|padding|>"
    hf_tokenizer.eos_token = "<|endoftext|>"
    hf_tokenizer.bos_token = "<|endoftext|>"
    hf_tokenizer.save_pretrained(save_dir)

    print(f"Tokenizer saved to {save_dir}")
    return hf_tokenizer


class TextDataset(Dataset):
    """Tokenize corpus into fixed-length chunks for causal LM training."""

    def __init__(self, corpus_path, tokenizer, seq_len):
        print(f"Tokenizing corpus: {corpus_path}")
        t0 = time.time()

        with open(corpus_path, encoding="utf-8") as f:
            text = f.read()

        all_ids = tokenizer.encode(text)

        # Naive contiguous chunking: no document boundaries, no EOS insertion.
        # The corpus is shuffled at sentence level upstream
        # (build_french_corpus.py), so within-chunk locality is already broken.
        n_chunks = len(all_ids) // seq_len
        self.chunks = []
        for i in range(n_chunks):
            start = i * seq_len
            self.chunks.append(all_ids[start:start + seq_len])

        elapsed = time.time() - t0
        print(f"  {len(all_ids):,} tokens -> {len(self.chunks):,} chunks of {seq_len} "
              f"({elapsed:.0f}s)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids}


def create_model(vocab_size):
    """Create a 125M GPT-2 model from scratch."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT-2 125M ({n_params/1e6:.1f}M parameters)")
    return model


def get_lr(step, warmup_steps, max_lr, total_steps):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, tokenizer, words_processed, models_dir):
    """Save model in HuggingFace format for eval pipeline compatibility."""
    # Both branches of the original if/else produced identical names; keep
    # the single int-floor formatting that matches the existing chck_NM
    # pattern on disk.
    millions = words_processed / 1_000_000
    name = f"chck_{int(millions)}M"

    save_dir = os.path.join(models_dir, name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {"words_processed": words_processed, "checkpoint_name": name}
    with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Checkpoint saved: {save_dir} ({words_processed/1e6:.1f}M words)")
    return save_dir


def estimate_words_per_token(corpus_path, tokenizer, sample_lines=10000):
    """Estimate the words-per-token ratio for checkpoint scheduling."""
    with open(corpus_path, encoding="utf-8") as f:
        lines = [f.readline() for _ in range(sample_lines)]
    text = "\n".join(lines)
    words = len(text.split())
    tokens = len(tokenizer.encode(text))
    ratio = words / tokens if tokens > 0 else 1.0
    print(f"Words/token ratio: {ratio:.3f} (sampled {sample_lines} lines)")
    return ratio


def train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device} | seed: {args.seed}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Tokenizer
    tokenizer = train_tokenizer(args.corpus, args.vocab_size, TOKENIZER_DIR)

    # Dataset
    dataset = TextDataset(args.corpus, tokenizer, args.seq_len)
    num_workers = 4 if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = GPT2LMHeadModel.from_pretrained(args.resume)
        meta_path = os.path.join(args.resume, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                words_processed = json.load(f)["words_processed"]
        else:
            words_processed = 0
    else:
        model = create_model(tokenizer.vocab_size)
        words_processed = 0

    model.to(device)

    # Compile for speed on CUDA
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Estimate words per token for checkpoint scheduling
    words_per_token = estimate_words_per_token(args.corpus, tokenizer)
    tokens_per_batch = args.batch_size * args.seq_len

    total_steps = len(dataloader) * args.epochs
    print(f"\nTraining: {args.epochs} epoch(s), {len(dataloader)} steps/epoch, "
          f"{total_steps} total steps")
    print(f"Checkpoint schedule: {[f'{w//1e6:.0f}M' for w in CHECKPOINT_WORDS]}")

    # Figure out which checkpoints we still need
    next_ckpt_idx = 0
    for i, w in enumerate(CHECKPOINT_WORDS):
        if w > words_processed:
            next_ckpt_idx = i
            break

    # Training loop
    model.train()
    global_step = 0
    log_interval = 100
    t0 = time.time()
    running_loss = 0.0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        for batch in dataloader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # LR schedule
            lr = get_lr(global_step, args.warmup_steps, args.lr, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            # Estimate words processed
            tokens_this_batch = input_ids.numel()
            words_processed += int(tokens_this_batch * words_per_token)

            # Logging
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                tokens_per_sec = (global_step * tokens_per_batch) / elapsed
                ppl = math.exp(min(avg_loss, 20))
                print(f"  step {global_step:>6} | loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                      f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                      f"{words_processed/1e6:.1f}M words")
                running_loss = 0.0

            # Checkpoint
            while (next_ckpt_idx < len(CHECKPOINT_WORDS)
                   and words_processed >= CHECKPOINT_WORDS[next_ckpt_idx]):
                save_checkpoint(model, tokenizer, words_processed, MODELS_DIR)
                next_ckpt_idx += 1

    # Final save
    save_checkpoint(model, tokenizer, words_processed, MODELS_DIR)
    print(f"\nTraining complete. {words_processed/1e6:.1f}M words processed.")
    print(f"Total time: {(time.time() - t0)/3600:.1f}h")
    print("\nTo evaluate: ./eval_zero_shot.sh models/chck_95M causal")


if __name__ == "__main__":
    args = parse_args()
    train(args)
