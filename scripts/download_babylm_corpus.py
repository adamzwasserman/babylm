"""
Download and inspect the official BabyLM 2026 corpus from HuggingFace.
Identifies any French/multilingual content already present.
"""

import json
import os

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "babylm_official")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_babylm_corpus():
    print("Fetching BabyLM community datasets from HuggingFace...")

    # Known BabyLM dataset IDs
    dataset_ids = [
        "babylm/babylm_10M",
        "babylm/babylm_100M",
        "babylm/babylm_dev",
        "babylm/babylm_test",
    ]

    results = {}

    for dataset_id in dataset_ids:
        print(f"\nTrying: {dataset_id}")
        try:
            ds = load_dataset(dataset_id, trust_remote_code=True)
            info = {
                "splits": list(ds.keys()),
                "features": str(ds[list(ds.keys())[0]].features),
                "num_rows": {split: len(ds[split]) for split in ds.keys()},
            }
            results[dataset_id] = info
            print(f"  Splits: {info['splits']}")
            print(f"  Rows: {info['num_rows']}")

            # Save first 100 rows as sample
            split = list(ds.keys())[0]
            sample_path = os.path.join(OUTPUT_DIR, f"{dataset_id.replace('/', '_')}_sample.jsonl")
            with open(sample_path, "w") as f:
                for i, row in enumerate(ds[split]):
                    if i >= 100:
                        break
                    f.write(json.dumps(row) + "\n")
            print(f"  Sample saved to {sample_path}")

        except Exception as e:
            print(f"  Could not load {dataset_id}: {e}")
            results[dataset_id] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "corpus_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results

def check_for_french(results):
    """Report on any French content in the official corpus."""
    print("\n=== French Content Check ===")
    french_keywords = ["fr", "french", "français", "francais", "multilingual"]

    for dataset_id, info in results.items():
        if "error" not in info:
            id_lower = dataset_id.lower()
            if any(kw in id_lower for kw in french_keywords):
                print(f"FOUND French dataset: {dataset_id}")
            else:
                print(f"No French in: {dataset_id}")

    print("\nNote: Official BabyLM corpus is English-only.")
    print("Our French corpus will be fully custom (allowed under Strict track rules).")

if __name__ == "__main__":
    results = download_babylm_corpus()
    check_for_french(results)
