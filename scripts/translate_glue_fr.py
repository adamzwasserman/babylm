"""
Translate GLUE train+valid data to French using Claude Sonnet API.
Preserves labels and structure, only translates text fields.

Usage:
  uv run python scripts/translate_glue_fr.py [--task boolq|rte|multirc|all]
"""

import json
import os
from pathlib import Path
import anthropic
import argparse

GLUE_DIR = Path("eval/evaluation-pipeline-2025/evaluation_data/full_eval/glue_filtered")
OUT_DIR = Path("eval/evaluation-pipeline-2025/evaluation_data/full_eval/glue_filtered_fr")

client = anthropic.Anthropic()

# Fields to translate per task
TASK_FIELDS = {
    "boolq":   ["passage", "question"],
    "mnli":    ["premise", "hypothesis"],
    "mrpc":    ["sentence1", "sentence2"],
    "multirc": ["paragraph", "question", "answer"],
    "qqp":     ["question1", "question2"],
    "rte":     ["sentence1", "sentence2"],
    "wsc":     ["text", "span1_text", "span2_text"],
}


def translate_texts(texts, batch_size=25):
    """Translate a list of English texts to French via Claude Haiku."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": f"""Translate each English text to natural French. Return ONLY numbered translations, one per line.

{numbered}"""
                }]
            )
            text = response.content[0].text.strip()
            batch_results = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(". ", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    batch_results.append(parts[1].strip())
                else:
                    batch_results.append(line)
            while len(batch_results) < len(batch):
                batch_results.append(batch[len(batch_results)])
            results.extend(batch_results[:len(batch)])
        except Exception as e:
            print(f"  API error: {e}", flush=True)
            results.extend(batch)
        if len(results) % 100 == 0 and len(results) > 0:
            print(f"  {len(results)}/{len(texts)} texts translated", flush=True)
    return results


def translate_task(task, split):
    """Translate one task split (train or valid)."""
    src = GLUE_DIR / f"{task}.{split}.jsonl"
    dst = OUT_DIR / f"{task}.{split}.jsonl"

    if dst.exists():
        print(f"  {task}.{split} (cached)", flush=True)
        return

    items = [json.loads(line) for line in open(src)]
    fields = TASK_FIELDS[task]

    # Collect all texts to translate
    all_texts = []
    field_map = []  # (item_idx, field_name)
    for idx, item in enumerate(items):
        for field in fields:
            all_texts.append(str(item[field]))
            field_map.append((idx, field))

    print(f"  {task}.{split}: {len(items)} items, {len(all_texts)} texts to translate", flush=True)

    # Translate
    translated = translate_texts(all_texts)

    # Reassemble
    for (idx, field), trans in zip(field_map, translated):
        items[idx][field] = trans

    # Write
    with open(dst, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  {task}.{split}: saved {len(items)} items", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = list(TASK_FIELDS.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        print(f"=== {task} ===", flush=True)
        translate_task(task, "train")
        translate_task(task, "valid")

    print("\nDone. French GLUE data in:", OUT_DIR)


if __name__ == "__main__":
    main()
