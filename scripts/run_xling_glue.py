"""Cross-lingual GLUE evaluation grid (§4.4 / Table 3).

Runs the four-lever experiment grid on a French-trained checkpoint:

    Baseline : rank-8 English LoRA, 3 epochs, on English task data
    A        : Baseline + 5 epochs (epoch-extension lever)
    B        : zero-shot inference-time FR-EN vocabulary axioms
               (the placebo control reported in §5.2; this lever is
               handled here so Table 3 can be assembled in one pass)
    C        : tuned rank-16 LoRA on English task data, alpha=32
    D+C      : "E3" - GLUE train/eval translated to French, then rank-16 LoRA

The five tasks are BoolQ, RTE, MRPC, WSC, and MNLI. LoRA leaves the French
base model bit-identical, so each (seed, lever, task) cell trains a fresh
adapter on top of the seed's checkpoint and evaluates the adapter only.

Usage:
    # one cell
    uv run python scripts/run_xling_glue.py models/seed42/chck_92M \\
        --seed 42 --lever C --task rte

    # all cells for one seed (five tasks x four FR-side levers, baseline
    # column is a separate run on the English reference model)
    uv run python scripts/run_xling_glue.py models/seed42/chck_92M \\
        --seed 42 --all_levers --all_tasks

Output:
    eval_results/seed{S}_xglue_<lever>_<task>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LEVERS = ("Baseline", "A", "B", "C", "D+C")
TASKS = ("boolq", "rte", "mrpc", "wsc", "mnli")


@dataclass
class TaskSpec:
    """How to load and score a GLUE-style task in both English and French."""
    en_dataset: str
    en_subset: str | None
    fr_dataset: str
    fr_subset: str | None
    text_a_field: str
    text_b_field: str | None
    label_field: str
    n_labels: int
    metric: str  # "accuracy" or "matthews"
    val_split: str = "validation"


# Task specifications. The fr_dataset entries point to the French-translated
# splits released with the submission (per §4.4 footnote, "translation script
# is released with the submission"). If the names below differ from the
# release, override via --fr_dataset_<task> CLI args (one per task).
TASK_SPECS: dict[str, TaskSpec] = {
    "boolq": TaskSpec(
        en_dataset="super_glue", en_subset="boolq",
        fr_dataset="submission/glue_fr", fr_subset="boolq",
        text_a_field="question", text_b_field="passage",
        label_field="label", n_labels=2, metric="accuracy",
    ),
    "rte": TaskSpec(
        en_dataset="super_glue", en_subset="rte",
        fr_dataset="submission/glue_fr", fr_subset="rte",
        text_a_field="premise", text_b_field="hypothesis",
        label_field="label", n_labels=2, metric="accuracy",
    ),
    "mrpc": TaskSpec(
        en_dataset="glue", en_subset="mrpc",
        fr_dataset="submission/glue_fr", fr_subset="mrpc",
        text_a_field="sentence1", text_b_field="sentence2",
        label_field="label", n_labels=2, metric="accuracy",
    ),
    "wsc": TaskSpec(
        en_dataset="super_glue", en_subset="wsc.fixed",
        fr_dataset="submission/glue_fr", fr_subset="wsc",
        text_a_field="text", text_b_field=None,
        label_field="label", n_labels=2, metric="accuracy",
    ),
    "mnli": TaskSpec(
        en_dataset="glue", en_subset="mnli",
        fr_dataset="submission/glue_fr", fr_subset="mnli",
        text_a_field="premise", text_b_field="hypothesis",
        label_field="label", n_labels=3, metric="accuracy",
        val_split="validation_matched",
    ),
}


@dataclass
class LeverSpec:
    """Per-lever knobs that control LoRA rank, training data language,
    epochs, and whether the adapter is trained at all."""
    rank: int | None  # None = no LoRA / zero-shot
    alpha: int | None
    epochs: float
    train_data_language: str  # "en" or "fr" or "none" (zero-shot)
    eval_data_language: str  # "en" or "fr"
    needs_training: bool


LEVER_SPECS: dict[str, LeverSpec] = {
    "Baseline": LeverSpec(rank=8, alpha=16, epochs=3.0,
                          train_data_language="en", eval_data_language="en",
                          needs_training=True),
    "A": LeverSpec(rank=8, alpha=16, epochs=5.0,
                   train_data_language="en", eval_data_language="en",
                   needs_training=True),
    # B is zero-shot inference-time vocab axioms; the actual axiom-prepending
    # lives in run_dict_axioms_placebo.py (§5.2), so here we record a stub
    # that points the aggregator to that file. Marked needs_training=False
    # so this script never accidentally fine-tunes it.
    "B": LeverSpec(rank=None, alpha=None, epochs=0.0,
                   train_data_language="none", eval_data_language="en",
                   needs_training=False),
    "C": LeverSpec(rank=16, alpha=32, epochs=3.0,
                   train_data_language="en", eval_data_language="en",
                   needs_training=True),
    "D+C": LeverSpec(rank=16, alpha=32, epochs=3.0,
                     train_data_language="fr", eval_data_language="fr",
                     needs_training=True),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def encode_pairs(examples, tokenizer, spec: TaskSpec, max_length: int):
    a = examples[spec.text_a_field]
    if spec.text_b_field is None:
        out = tokenizer(a, truncation=True, max_length=max_length)
    else:
        b = examples[spec.text_b_field]
        out = tokenizer(a, b, truncation=True, max_length=max_length)
    out["labels"] = examples[spec.label_field]
    return out


def load_task_split(spec: TaskSpec, language: str, split: str,
                    fr_dataset_override: str | None = None,
                    fr_subset_override: str | None = None):
    if language == "en":
        if spec.en_subset:
            return load_dataset(spec.en_dataset, spec.en_subset, split=split)
        return load_dataset(spec.en_dataset, split=split)

    fr_ds = fr_dataset_override or spec.fr_dataset
    fr_sub = fr_subset_override or spec.fr_subset

    # Local directory of {task}.{train,valid}.jsonl files (the FR-translated
    # GLUE shipped under submission/glue_fr/). Multiple HF validation split
    # names (validation, validation_matched, validation_mismatched) all map
    # to the single local valid.jsonl.
    fr_ds_path = Path(fr_ds)
    if not fr_ds_path.is_absolute():
        fr_ds_path = _project_root() / fr_ds_path
    if fr_ds_path.is_dir():
        local_split = "train" if split == "train" else "valid"
        data_file = fr_ds_path / f"{fr_sub}.{local_split}.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Expected translated GLUE file {data_file} (task={fr_sub}, "
                f"split={split} -> {local_split})"
            )
        return load_dataset("json", data_files=str(data_file), split="train")

    if fr_sub:
        return load_dataset(fr_ds, fr_sub, split=split)
    return load_dataset(fr_ds, split=split)


def compute_metrics_factory(spec: TaskSpec):
    def _compute(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        out = {"accuracy": accuracy_score(labels, preds)}
        if spec.n_labels == 2:
            out["f1"] = f1_score(labels, preds, average="binary", zero_division=0)
        if spec.metric == "matthews":
            out["mcc"] = matthews_corrcoef(labels, preds)
        return out
    return _compute


def run_one_cell(checkpoint: str, lever: str, task: str, seed: int | None,
                 batch_size: int, lr: float, max_length: int,
                 fr_dataset_override: str | None,
                 fr_subset_override: str | None) -> dict:
    spec = TASK_SPECS[task]
    lspec = LEVER_SPECS[lever]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_arg = seed if seed is not None else 42

    if not lspec.needs_training:
        return {
            "checkpoint": checkpoint, "seed": seed, "lever": lever, "task": task,
            "skipped": True,
            "reason": "lever B is zero-shot vocabulary-axiom prompting; see "
                      "scripts/run_dict_axioms_placebo.py for the actual evaluation.",
        }

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=spec.n_labels,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if lspec.rank is not None:
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lspec.rank,
            lora_alpha=lspec.alpha,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base, peft_cfg)
        model.print_trainable_parameters()
    else:
        model = base
    model.to(device)

    train_split = "train"
    val_split = spec.val_split
    print(f"Loading {task} train ({lspec.train_data_language})...")
    train_raw = load_task_split(
        spec, lspec.train_data_language, train_split,
        fr_dataset_override, fr_subset_override,
    )
    print(f"Loading {task} eval ({lspec.eval_data_language})...")
    val_raw = load_task_split(
        spec, lspec.eval_data_language, val_split,
        fr_dataset_override, fr_subset_override,
    )

    def _tok(b):
        return encode_pairs(b, tokenizer, spec, max_length)

    keep_after_tok = [spec.label_field]
    train_ds = train_raw.map(_tok, batched=True,
                              remove_columns=[c for c in train_raw.column_names
                                              if c not in keep_after_tok])
    val_ds = val_raw.map(_tok, batched=True,
                          remove_columns=[c for c in val_raw.column_names
                                          if c not in keep_after_tok])

    output_dir = _project_root() / "eval_results" / f"_xglue_seed{seed_arg}_{lever}_{task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=lspec.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
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
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_factory(spec),
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")

    return {
        "checkpoint": checkpoint,
        "seed": seed,
        "lever": lever,
        "task": task,
        "metrics": {
            k.replace("eval_", ""): float(v)
            for k, v in eval_metrics.items()
            if isinstance(v, (int, float))
        },
        "lora": (
            {"rank": lspec.rank, "alpha": lspec.alpha, "epochs": lspec.epochs}
            if lspec.rank is not None else None
        ),
        "train_data_language": lspec.train_data_language,
        "eval_data_language": lspec.eval_data_language,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="French model checkpoint")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the output filenames")
    p.add_argument("--lever", choices=list(LEVERS) + ["all"], default="all")
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    p.add_argument("--all_levers", action="store_true")
    p.add_argument("--all_tasks", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--fr_dataset", default=None,
                   help="Override the HF dataset id used for French task data "
                        "(applies to all tasks)")
    p.add_argument("--fr_subset", default=None,
                   help="Override the HF subset used for French task data")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.output_dir) if args.output_dir else _project_root() / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    levers = list(LEVERS) if (args.lever == "all" or args.all_levers) else [args.lever]
    tasks = list(TASKS) if (args.task == "all" or args.all_tasks) else [args.task]

    seed_tag = f"seed{args.seed}" if args.seed is not None else Path(args.checkpoint).name

    for lever in levers:
        for task in tasks:
            print(f"\n=== {seed_tag} | lever={lever} | task={task} ===")
            res = run_one_cell(
                args.checkpoint, lever, task, args.seed,
                args.batch_size, args.lr, args.max_length,
                args.fr_dataset, args.fr_subset,
            )
            out_path = out_dir / f"{seed_tag}_xglue_{lever.replace('+', '')}_{task}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
