"""Build the multi-seed paper Tables 1, 2, and 3 from per-seed eval JSONs.

Reads:
    eval_results/seed{S}_qfrblimp.json   -> Table 1 (QFrBLiMP buckets + overall)
    eval_results/seed{S}_qfrcola.json    -> §4.1 prose (QFrCoLA acc + MCC)
    eval_results/seed{S}_babylm.json     -> §4.2 prose (BLiMP / BLiMP-Sup / EWoK / GLUE-mean)
    eval_results/seed{S}_bli_<target>.json -> Table 2 (BLI Procrustes)
    eval_results/seed{S}_xglue_<lever>_<task>.json -> Table 3 (cross-lingual GLUE)

Writes one LaTeX `tabular` block per table (mean +/- std across seeds), plus
a markdown summary. The aggregator does not modify the LaTeX of paper/, it
only emits the numbers; copy-paste into the paper or tee to a target file.

Usage:
    uv run python scripts/aggregate_paper_tables.py
    uv run python scripts/aggregate_paper_tables.py --eval_dir eval_results
    uv run python scripts/aggregate_paper_tables.py --output paper_tables.tex
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = PROJECT_ROOT / "eval_results"

SEED_RE = re.compile(r"seed(\d+)")


def fmt(values: list[float], pct: bool = True, digits: int = 2) -> str:
    """Format mean +/- std for a list of seed-level values."""
    if not values:
        return "-"
    if len(values) == 1:
        v = values[0] * 100 if pct else values[0]
        return f"${v:.{digits}f}$" if not pct else f"${v:.{digits}f}\\%$"
    m = mean(values)
    s = stdev(values)
    if pct:
        return f"${m * 100:.{digits}f} \\pm {s * 100:.{digits}f}$"
    return f"${m:.{digits}f} \\pm {s:.{digits}f}$"


def fmt_md(values: list[float], pct: bool = True, digits: int = 2) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        v = values[0] * 100 if pct else values[0]
        return f"{v:.{digits}f}" + ("%" if pct else "")
    m = mean(values)
    s = stdev(values)
    if pct:
        return f"{m * 100:.{digits}f}% +/- {s * 100:.{digits}f}"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def _seed_from_filename(path: str) -> int | None:
    m = SEED_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def collect(pattern: str, eval_dir: Path) -> list[tuple[int | None, dict]]:
    files = sorted(glob.glob(str(eval_dir / pattern)))
    out: list[tuple[int | None, dict]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                print(f"WARN: skipping malformed JSON {f}", file=sys.stderr)
                continue
        seed = data.get("seed") if isinstance(data, dict) else None
        if seed is None:
            seed = _seed_from_filename(f)
        out.append((seed, data))
    return out


# ---- Table 1: QFrBLiMP ----------------------------------------------------


BUCKETS_T1 = ("syntactic", "semantic", "morphological", "anglicism_related")


def table1_qfrblimp(eval_dir: Path) -> tuple[str, str]:
    rows = collect("seed*_qfrblimp.json", eval_dir)
    if not rows:
        return "", "(no QFrBLiMP results found)"

    bucket_values: dict[str, list[float]] = defaultdict(list)
    overall_values: list[float] = []
    for _seed, data in rows:
        overall_values.append(data["overall"])
        for b in BUCKETS_T1:
            v = data["buckets"].get(b)
            if v is not None:
                bucket_values[b].append(v)

    md_lines = [
        "## Table 1 — QFrBLiMP (mean ± std across seeds)",
        "",
        "| Bucket | Accuracy |",
        "|---|---|",
    ]
    for b in BUCKETS_T1:
        md_lines.append(f"| {b.replace('_', ' ').title()} | {fmt_md(bucket_values[b])} |")
    md_lines.append(f"| **Overall** | **{fmt_md(overall_values)}** |")
    md_lines.append(f"| n seeds | {len(rows)} |")

    tex = [
        "% Table 1: QFrBLiMP buckets, mean +/- std across seeds",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Bucket & Accuracy \\",
        r"\midrule",
    ]
    for b in BUCKETS_T1:
        tex.append(f"{b.replace('_', ' ').title()} & {fmt(bucket_values[b])} \\\\")
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Overall}} & \\textbf{{{fmt(overall_values)}}} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    return "\n".join(tex), "\n".join(md_lines)


# ---- §4.1 prose: QFrBLiMP multi-epoch trajectory --------------------------


_EPOCH_PAT = re.compile(r"seed(\d+)_qfrblimp_epoch(\d+)\.json$")


def trajectory_qfrblimp(eval_dir: Path) -> str:
    """Per-seed, per-epoch QFrBLiMP overall accuracy. Mirrors the
    multi-epoch trajectory the paper reports in §4.1 prose."""
    files = sorted(glob.glob(str(eval_dir / "seed*_qfrblimp_epoch*.json")))
    if not files:
        return "(no per-epoch QFrBLiMP results)\n"
    by_epoch: dict[int, dict[int, float]] = defaultdict(dict)
    seeds: set[int] = set()
    for f in files:
        m = _EPOCH_PAT.search(Path(f).name)
        if not m:
            continue
        seed = int(m.group(1))
        epoch = int(m.group(2))
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        score = data.get("overall")
        if score is None:
            continue
        by_epoch[epoch][seed] = float(score)
        seeds.add(seed)
    if not by_epoch:
        return "(no per-epoch QFrBLiMP results)\n"

    best_epoch_per_seed: dict[int, int] = {}
    for s in seeds:
        bp = eval_dir / f"seed{s}_best_epoch.json"
        if bp.exists():
            with open(bp, encoding="utf-8") as fh:
                d = json.load(fh)
                if "best_epoch" in d:
                    best_epoch_per_seed[s] = int(d["best_epoch"])

    seeds_sorted = sorted(seeds)
    epochs_sorted = sorted(by_epoch.keys())
    lines = [
        "## §4.1 prose — QFrBLiMP multi-epoch trajectory",
        "",
        "Overall accuracy per seed (rows = epochs).",
        "Per-seed best epoch is marked with *; that is the row picked for Table 1.",
        "",
        "| Epoch | " + " | ".join(f"seed {s}" for s in seeds_sorted) + " | mean +/- std |",
        "|" + "---|" * (len(seeds_sorted) + 2),
    ]
    for epoch in epochs_sorted:
        cells: list[str] = []
        vals: list[float] = []
        for s in seeds_sorted:
            v = by_epoch[epoch].get(s)
            if v is None:
                cells.append("-")
            else:
                marker = "*" if best_epoch_per_seed.get(s) == epoch else ""
                cells.append(f"{v * 100:.2f}{marker}")
                vals.append(v)
        cells.append(fmt_md(vals) if vals else "-")
        lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---- §4.1 prose: QFrCoLA --------------------------------------------------


def prose_qfrcola(eval_dir: Path) -> str:
    rows = collect("seed*_qfrcola.json", eval_dir)
    if not rows:
        return "(no QFrCoLA results found)\n"
    in_acc, in_mcc = [], []
    for _seed, d in rows:
        if "in_domain_test" in d:
            in_acc.append(d["in_domain_test"]["accuracy"])
            in_mcc.append(d["in_domain_test"]["mcc"])
    return (
        "## §4.1 prose — QFrCoLA (mean +/- std across seeds)\n"
        f"  in-domain test accuracy : {fmt_md(in_acc)}\n"
        f"  in-domain test MCC      : {fmt_md(in_mcc, pct=False, digits=3)}\n"
        f"  n seeds                 : {len(rows)}\n"
    )


# ---- §4.2 prose: BabyLM weighted leaderboard ------------------------------


def _avg_from_babylm_block(block: dict | None) -> float | None:
    if not isinstance(block, dict):
        return None
    if "average" in block and isinstance(block["average"], (int, float)):
        return float(block["average"])
    # Some pipeline outputs nest under "results" or "score"; pick the first
    # numeric leaf at the top level.
    for _k, v in block.items():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def prose_babylm(eval_dir: Path) -> str:
    rows = collect("seed*_babylm.json", eval_dir)
    if not rows:
        return "(no BabyLM results found)\n"
    blimp, supp, ewok, glue = [], [], [], []
    for _seed, d in rows:
        for key, bucket in (("blimp", blimp), ("blimp_supplement", supp),
                             ("ewok", ewok), ("glue", glue)):
            v = _avg_from_babylm_block(d.get(key))
            if v is not None:
                bucket.append(v)
    return (
        "## §4.2 prose — BabyLM weighted leaderboard (mean +/- std)\n"
        f"  BLiMP            : {fmt_md(blimp)}\n"
        f"  BLiMP Supplement : {fmt_md(supp)}\n"
        f"  EWoK             : {fmt_md(ewok)}\n"
        f"  GLUE (mean)      : {fmt_md(glue)}\n"
        f"  n seeds          : {len(rows)}\n"
    )


# ---- Table 2: BLI Procrustes ----------------------------------------------


def table2_bli(eval_dir: Path) -> tuple[str, str]:
    rows = collect("seed*_bli_*.json", eval_dir)
    if not rows:
        return "", "(no BLI results found)"

    by_target: dict[str, list[dict]] = defaultdict(list)
    for _seed, d in rows:
        by_target[d.get("target_label", "unknown")].append(d)

    keys = ("fit", "p@1", "p@5", "p@10")

    md = ["## Table 2 — BLI Procrustes (mean +/- std across seeds)",
          "",
          "| Target | Fit | p@1 | p@5 | p@10 | n |",
          "|---|---|---|---|---|---|"]
    tex = [
        "% Table 2: BLI Procrustes alignment, mean +/- std across seeds",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Target & Fit & p@1 & p@5 & p@10 \\",
        r"\midrule",
    ]
    for target, dl in sorted(by_target.items()):
        cells_tex, cells_md = [], []
        for k in keys:
            vals = [d[k] for d in dl if k in d]
            cells_tex.append(fmt(vals, pct=(k != "fit"), digits=2 if k == "fit" else 1))
            cells_md.append(fmt_md(vals, pct=(k != "fit"), digits=2 if k == "fit" else 1))
        tex.append(f"{target} & " + " & ".join(cells_tex) + r" \\")
        md.append(f"| {target} | " + " | ".join(cells_md) + f" | {len(dl)} |")

    # Random-orthogonal and chance baselines (taken from the first run that
    # carries them; they are deterministic per seed but reported as a
    # reference, not a per-seed average).
    if rows:
        any_d = rows[0][1]
        rnd = any_d.get("random_orthogonal")
        if rnd:
            tex.append("Random orthogonal & "
                       f"{rnd['fit']:.2f} & {rnd['p@1'] * 100:.1f} & "
                       f"{rnd['p@5'] * 100:.1f} & {rnd['p@10'] * 100:.1f} \\\\")
            md.append(f"| Random orthogonal (ref) | {rnd['fit']:.2f} | "
                       f"{rnd['p@1'] * 100:.1f}% | {rnd['p@5'] * 100:.1f}% | "
                       f"{rnd['p@10'] * 100:.1f}% | - |")
        chance = any_d.get("chance")
        if chance:
            tex.append(f"Chance & - & {chance['1'] * 100:.1f if isinstance(chance['1'], float) else chance.get(1, 0) * 100:.1f} & "
                       f"{chance.get(5, 0) * 100:.1f} & {chance.get(10, 0) * 100:.1f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    return "\n".join(tex), "\n".join(md)


# ---- Table 3: Cross-lingual GLUE ------------------------------------------


XLEVER_ORDER = ("Baseline", "A", "B", "C", "D+C")
TASK_ORDER = ("boolq", "rte", "mrpc", "wsc", "mnli")


def _xlever_from_filename(name: str) -> str | None:
    # Filename pattern: seed{S}_xglue_<lever>_<task>.json
    # Note: '+' was stripped at write time, so we re-canonicalise.
    parts = name.split("_xglue_", 1)
    if len(parts) < 2:
        return None
    rest = parts[1].rsplit(".json", 1)[0]
    pieces = rest.rsplit("_", 1)
    if len(pieces) != 2:
        return None
    lever = pieces[0].replace("DC", "D+C")  # invert the strip done in writer
    return lever


def _xtask_from_filename(name: str) -> str | None:
    parts = name.rsplit("_", 1)
    if len(parts) < 2:
        return None
    return parts[1].rsplit(".json", 1)[0]


def _accuracy_of(d: dict) -> float | None:
    m = d.get("metrics")
    if not isinstance(m, dict):
        return None
    return m.get("accuracy")


def table3_xling_glue(eval_dir: Path) -> tuple[str, str]:
    files = sorted(glob.glob(str(eval_dir / "seed*_xglue_*_*.json")))
    if not files:
        return "", "(no cross-lingual GLUE results found)"

    # cell[task][lever] = list of per-seed accuracies
    cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        name = Path(f).name
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("skipped"):
            continue
        lever = d.get("lever") or _xlever_from_filename(name)
        task = d.get("task") or _xtask_from_filename(name)
        acc = _accuracy_of(d)
        if lever and task and acc is not None:
            cell[task][lever].append(acc)

    md = ["## Table 3 — Cross-lingual GLUE (mean +/- std across seeds)",
          "",
          "| Task | " + " | ".join(XLEVER_ORDER) + " | Δ (D+C - Baseline) |",
          "|" + "---|" * (len(XLEVER_ORDER) + 2)]
    tex = [
        "% Table 3: Cross-lingual GLUE LoRA grid, mean +/- std across seeds",
        r"\begin{tabular}{l" + "c" * len(XLEVER_ORDER) + "c}",
        r"\toprule",
        "Task & " + " & ".join(XLEVER_ORDER) + r" & $\Delta$ \\",
        r"\midrule",
    ]
    means_per_lever: dict[str, list[float]] = defaultdict(list)
    for task in TASK_ORDER:
        if task not in cell:
            continue
        cells_tex, cells_md = [], []
        for lev in XLEVER_ORDER:
            vals = cell[task].get(lev, [])
            cells_tex.append(fmt(vals))
            cells_md.append(fmt_md(vals))
            if vals:
                means_per_lever[lev].append(mean(vals))
        base_vals = cell[task].get("Baseline", [])
        dc_vals = cell[task].get("D+C", [])
        if base_vals and dc_vals:
            delta = mean(dc_vals) - mean(base_vals)
            cells_tex.append(f"{delta * 100:+.2f}")
            cells_md.append(f"{delta * 100:+.2f}pp")
        else:
            cells_tex.append("-")
            cells_md.append("-")
        tex.append(f"{task.upper()} & " + " & ".join(cells_tex) + r" \\")
        md.append(f"| {task.upper()} | " + " | ".join(cells_md) + " |")

    if means_per_lever:
        tex.append(r"\midrule")
        row_tex, row_md = ["Mean"], ["**Mean**"]
        for lev in XLEVER_ORDER:
            ms = means_per_lever.get(lev, [])
            row_tex.append(f"{mean(ms) * 100:.2f}" if ms else "-")
            row_md.append(f"{mean(ms) * 100:.2f}%" if ms else "-")
        base_means = means_per_lever.get("Baseline", [])
        dc_means = means_per_lever.get("D+C", [])
        if base_means and dc_means:
            d = mean(dc_means) - mean(base_means)
            row_tex.append(f"{d * 100:+.2f}")
            row_md.append(f"{d * 100:+.2f}pp")
        else:
            row_tex.append("-")
            row_md.append("-")
        tex.append(" & ".join(row_tex) + r" \\")
        md.append("| " + " | ".join(row_md) + " |")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    return "\n".join(tex), "\n".join(md)


# ---- main ------------------------------------------------------------------


def write_block(parts: Iterable[str], path: Path) -> None:
    text = "\n\n".join(p for p in parts if p)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", default=str(DEFAULT_EVAL_DIR),
                   help="Directory holding per-seed eval JSONs")
    p.add_argument("--output", default=None,
                   help="Path to write the LaTeX block (default: stdout)")
    p.add_argument("--md_output", default=None,
                   help="Path to write the markdown summary (default: stdout)")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        sys.exit(f"eval directory not found: {eval_dir}")

    t1_tex, t1_md = table1_qfrblimp(eval_dir)
    traj_md = trajectory_qfrblimp(eval_dir)
    t2_tex, t2_md = table2_bli(eval_dir)
    t3_tex, t3_md = table3_xling_glue(eval_dir)
    qcola_md = prose_qfrcola(eval_dir)
    babylm_md = prose_babylm(eval_dir)

    tex_block = "\n\n".join(b for b in (t1_tex, t2_tex, t3_tex) if b)
    md_block = "\n\n".join(
        b for b in (t1_md, traj_md, qcola_md, babylm_md, t2_md, t3_md) if b
    )

    if args.output:
        Path(args.output).write_text(tex_block + "\n", encoding="utf-8")
        print(f"Wrote LaTeX -> {args.output}")
    else:
        print("\n=== LaTeX ===\n")
        print(tex_block)

    if args.md_output:
        Path(args.md_output).write_text(md_block + "\n", encoding="utf-8")
        print(f"Wrote markdown -> {args.md_output}")
    else:
        print("\n=== Markdown summary ===\n")
        print(md_block)


if __name__ == "__main__":
    main()
