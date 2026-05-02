"""
Word counter for BabyLM corpus validation.
BabyLM uses simple whitespace-split word counting (not subword tokens).
Run this on any text file or directory to check against the 100M word budget.
"""

import argparse
import os
import sys


def count_words_file(path):
    """Count whitespace-split words in a single file.

    Raises OSError on read failure rather than silently returning 0,
    which would otherwise let a corrupted or missing file pass the
    BabyLM budget check by undercounting.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(len(line.split()) for line in f)

def count_words_dir(directory, extensions=(".txt", ".json", ".jsonl")):
    """Recursively count words in all text files in a directory.

    Files that fail to read are reported on stderr and skipped, but
    the directory traversal does not exit silently. Returns the sum
    over readable files plus the list of (path, count) entries; the
    caller can compare entry count to expected file count to detect
    skipped files.
    """
    total = 0
    file_counts = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fname in files:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            path = os.path.join(root, fname)
            try:
                count = count_words_file(path)
            except OSError as e:
                print(f"  SKIP {path}: {e}", file=sys.stderr)
                continue
            file_counts.append((path, count))
            total += count

    return total, file_counts

def format_count(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def main():
    parser = argparse.ArgumentParser(description="Count words for BabyLM budget tracking")
    parser.add_argument("path", help="File or directory to count")
    parser.add_argument("--budget", type=int, default=100_000_000, help="Word budget (default: 100M)")
    parser.add_argument("--verbose", action="store_true", help="Show per-file counts")
    args = parser.parse_args()

    BUDGET = args.budget
    STRICT_SMALL_BUDGET = 10_000_000

    if os.path.isfile(args.path):
        count = count_words_file(args.path)
        print(f"File: {args.path}")
        print(f"Words: {format_count(count)} ({count:,})")
        file_counts = [(args.path, count)]
        total = count
    elif os.path.isdir(args.path):
        total, file_counts = count_words_dir(args.path)
        if args.verbose:
            print("\nPer-file breakdown:")
            for path, count in sorted(file_counts, key=lambda x: -x[1]):
                print(f"  {format_count(count):>8}  {path}")
        print(f"\nDirectory: {args.path}")
        print(f"Files: {len(file_counts)}")
    else:
        print(f"ERROR: {args.path} not found")
        sys.exit(1)

    print("\n=== BabyLM Budget Status ===")
    print(f"Total words:         {format_count(total):>10} ({total:,})")
    print(f"Strict budget:       {format_count(BUDGET):>10} ({BUDGET:,})")
    print(f"Strict-Small budget: {format_count(STRICT_SMALL_BUDGET):>10} ({STRICT_SMALL_BUDGET:,})")
    print("")

    strict_pct = total / BUDGET * 100
    small_pct = total / STRICT_SMALL_BUDGET * 100

    print(f"Strict track:        {strict_pct:>9.1f}% of budget used")
    print(f"Strict-Small track:  {small_pct:>9.1f}% of budget used")

    if total > BUDGET:
        over = total - BUDGET
        print(f"\nWARNING: Over Strict budget by {format_count(over)} words!")
    elif total > STRICT_SMALL_BUDGET:
        remaining = BUDGET - total
        print(f"\nStatus: Within Strict budget. {format_count(remaining)} words remaining.")
        print("        Exceeds Strict-Small budget (10M).")
    else:
        remaining_strict = BUDGET - total
        remaining_small = STRICT_SMALL_BUDGET - total
        print("\nStatus: Within BOTH budgets.")
        print(f"        Strict remaining:       {format_count(remaining_strict)}")
        print(f"        Strict-Small remaining: {format_count(remaining_small)}")

if __name__ == "__main__":
    main()
