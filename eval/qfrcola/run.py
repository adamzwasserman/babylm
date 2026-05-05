"""QFrCoLA fine-tuning + evaluation harness.

QFrCoLA (Beauchemin and Khoury, 2025) is a sentence-level acceptability
judgment corpus for Quebec French, with ~25,153 in-domain sentences and a
2,675-sentence out-of-domain split. We fine-tune a classification head on
top of the GPT-2-style French model and report:

    - in-domain test accuracy
    - in-domain test Matthews Correlation Coefficient (MCC, the standard
      metric for binary acceptability under class imbalance)
    - in-domain dev MCC (for early-stopping picking)
    - out-of-domain accuracy and MCC

Usage:
    uv run python eval/qfrcola/run.py models/seed42/chck_92M --seed 42

Output:
    eval_results/seed{S}_qfrcola.json with the metric block.

Notes:
    - The base model is loaded with AutoModelForSequenceClassification, which
      adds a randomly-initialised classification head; the rest of the model
      is fine-tuned end-to-end as in the paper (no LoRA here, fine-tune is
      cheap at this scale).
    - Random seeds for the fine-tune are derived from --seed so that ablation
      seeds map deterministically to fine-tuning seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATASET = "graalul/qfrcola"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def tokenize_dataset(ds, tokenizer, sentence_field, label_field, max_length):
    def _tok(batch):
        out = tokenizer(
            batch[sentence_field],
            truncation=True,
            max_length=max_length,
        )
        out["labels"] = batch[label_field]
        return out

    cols_to_drop = [c for c in ds.column_names if c not in {label_field, sentence_field}]
    cols_to_drop.append(sentence_field)
    if label_field != "labels":
        cols_to_drop.append(label_field)
    return ds.map(_tok, batched=True, remove_columns=cols_to_drop)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds),
    }


def evaluate(checkpoint: str, dataset_name: str, seed: int | None,
             sentence_field: str, label_field: str,
             ood_split_name: str | None,
             num_train_epochs: float, batch_size: int, lr: float,
             max_length: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        # The project's BPE tokenizer is trained without special tokens, so
        # eos/bos may also be None on a fresh checkpoint. Fall back through
        # the available special tokens, or register a [PAD] if everything is
        # missing — the embedding row is added to the model below.
        fallback = (tokenizer.eos_token
                    or tokenizer.bos_token
                    or tokenizer.unk_token)
        if fallback is not None:
            tokenizer.pad_token = fallback
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2,
    )
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    print(f"Loading {dataset_name}...")
    raw = load_dataset(dataset_name)
    print(f"  splits: {list(raw.keys())}")

    train_ds = tokenize_dataset(raw["train"], tokenizer, sentence_field, label_field, max_length)
    val_split = "validation" if "validation" in raw else "dev" if "dev" in raw else None
    val_ds = tokenize_dataset(
        raw[val_split], tokenizer, sentence_field, label_field, max_length,
    ) if val_split else None
    test_ds = tokenize_dataset(raw["test"], tokenizer, sentence_field, label_field, max_length)

    seed_arg = seed if seed is not None else 42
    project_root = _project_root()
    output_dir = project_root / "eval_results" / f"_qfrcola_train_seed{seed_arg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="no",
        logging_steps=200,
        seed=seed_arg,
        fp16=device.type == "cuda",
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = {"checkpoint": checkpoint, "seed": seed, "dataset": dataset_name}
    if val_ds is not None:
        dev_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="dev")
        metrics["in_domain_dev"] = {
            "accuracy": float(dev_metrics["dev_accuracy"]),
            "mcc": float(dev_metrics["dev_mcc"]),
        }

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    metrics["in_domain_test"] = {
        "accuracy": float(test_metrics["test_accuracy"]),
        "mcc": float(test_metrics["test_mcc"]),
    }

    if ood_split_name and ood_split_name in raw:
        ood_ds = tokenize_dataset(
            raw[ood_split_name], tokenizer, sentence_field, label_field, max_length,
        )
        ood_metrics = trainer.evaluate(eval_dataset=ood_ds, metric_key_prefix="ood")
        metrics["out_of_domain"] = {
            "accuracy": float(ood_metrics["ood_accuracy"]),
            "mcc": float(ood_metrics["ood_mcc"]),
        }

    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a HuggingFace checkpoint dir")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the output filename and HF Trainer seed")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--sentence_field", default="sentence")
    p.add_argument("--label_field", default="label")
    p.add_argument("--ood_split", default="ood",
                   help="Name of the out-of-domain split (skipped if absent)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.output_dir) if args.output_dir else _project_root() / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        args.checkpoint, args.dataset, args.seed,
        args.sentence_field, args.label_field, args.ood_split,
        args.epochs, args.batch_size, args.lr, args.max_length,
    )

    seed_tag = f"seed{args.seed}" if args.seed is not None else Path(args.checkpoint).name
    out_path = out_dir / f"{seed_tag}_qfrcola.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nQFrCoLA results:")
    if "in_domain_test" in result:
        print(f"  in-domain test : acc={result['in_domain_test']['accuracy']:.4f}"
              f" mcc={result['in_domain_test']['mcc']:.4f}")
    if "in_domain_dev" in result:
        print(f"  in-domain dev  : acc={result['in_domain_dev']['accuracy']:.4f}"
              f" mcc={result['in_domain_dev']['mcc']:.4f}")
    if "out_of_domain" in result:
        print(f"  out-of-domain  : acc={result['out_of_domain']['accuracy']:.4f}"
              f" mcc={result['out_of_domain']['mcc']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
