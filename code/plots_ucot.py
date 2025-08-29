# plots_ucot.py
# Make bar charts from U-CoT behavior summaries (no seaborn).
# Input : artifacts/.../behavior_eval*.json  (from your strict evaluator)
# Output: PNGs + CSV + a short markdown snippet.

import json, argparse, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

METRICS = ["accuracy", "hint_follow", "deny_but_use", "admit_but_ignore", "probe_acc"]
ORDER   = ["helpful", "misleading", "irrelevant", "overall"]  # plot first 3 grouped; overall separate

def load_summary(path: Path):
    J = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in J:
        raise ValueError(f"No 'summary' in {path}")
    return J["summary"]

def safe_get(s, k):
    v = s.get(k, float("nan"))
    # allow strings "nan" etc.
    try: return float(v)
    except: return float("nan")

def bar_group(ax, groups, series, title, ylabel):
    # groups e.g. ["helpful","misleading","irrelevant"]
    # series : dict metric_name -> list of values per group
    x = np.arange(len(groups))
    n = len(series)
    width = 0.18 if n>=4 else 0.22
    offsets = (np.arange(n) - (n-1)/2) * (width + 0.02)

    for i, (name, vals) in enumerate(series.items()):
        y = np.array(vals, dtype=float)
        ax.bar(x + offsets[i], y, width, label=name)
        # annotate
        for xi, yi in zip(x + offsets[i], y):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="behavior_eval_v*.json from step4 strict evaluator")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    p = Path(args.in_json)
    out = Path(args.out_dir) if args.out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    S = load_summary(p)

    # assemble rows for CSV + plots
    rows = []
    for k in ORDER:
        if k not in S: continue
        row = {"group": k, "n": int(S[k].get("n", 0))}
        for m in METRICS:
            row[m] = safe_get(S[k], m)
        rows.append(row)

    # write CSV
    csv_path = out / "ucot_behavior_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group","n"] + METRICS)
        w.writeheader()
        for r in rows: w.writerow(r)
    print("Wrote:", csv_path)

    # Figure 1: grouped bars for per-hint metrics (helpful/misleading/irrelevant)
    groups = [r["group"] for r in rows if r["group"] in ("helpful","misleading","irrelevant")]
    series = {
        "accuracy":      [r["accuracy"]      for r in rows if r["group"] in ("helpful","misleading","irrelevant")],
        "hint_follow":   [r["hint_follow"]   for r in rows if r["group"] in ("helpful","misleading","irrelevant")],
        "deny_but_use":  [r["deny_but_use"]  for r in rows if r["group"] in ("helpful","misleading","irrelevant")],
        "admit_ignore":  [r["admit_but_ignore"] for r in rows if r["group"] in ("helpful","misleading","irrelevant")],
    }

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    bar_group(ax, groups, series, title="U-CoT metrics by hint type", ylabel="rate")
    plt.tight_layout()
    plt.savefig(out / "ucot_metrics_by_hint.png", dpi=220)
    plt.close()
    print("Wrote:", out / "ucot_metrics_by_hint.png")

    # Figure 2: overall vs per-hint accuracy (simple bars)
    acc_vals = [r["accuracy"] for r in rows if r["group"] in ("helpful","misleading","irrelevant","overall")]
    acc_labels= [r["group"] for r in rows if r["group"] in ("helpful","misleading","irrelevant","overall")]
    plt.figure(figsize=(6,4))
    x = np.arange(len(acc_vals))
    plt.bar(x, acc_vals, width=0.6)
    for xi, yi, lbl in zip(x, acc_vals, acc_labels):
        if np.isfinite(yi):
            plt.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, acc_labels)
    plt.ylim(0, 1.05)
    plt.title("Answer accuracy")
    plt.ylabel("accuracy")
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / "ucot_accuracy.png", dpi=220)
    plt.close()
    print("Wrote:", out / "ucot_accuracy.png")

    # Figure 3: probe accuracy per group (if present)
    if any(np.isfinite(r["probe_acc"]) for r in rows):
        probe_vals = [r["probe_acc"] for r in rows if r["group"] in ("helpful","misleading","irrelevant","overall")]
        probe_labels= [r["group"]     for r in rows if r["group"] in ("helpful","misleading","irrelevant","overall")]
        plt.figure(figsize=(6,4))
        x = np.arange(len(probe_vals))
        plt.bar(x, probe_vals, width=0.6)
        for xi, yi in zip(x, probe_vals):
            if np.isfinite(yi):
                plt.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(x, probe_labels)
        plt.ylim(0, 1.05)
        plt.title("Assistant-start probe accuracy")
        plt.ylabel("accuracy")
        plt.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out / "ucot_probe_acc.png", dpi=220)
        plt.close()
        print("Wrote:", out / "ucot_probe_acc.png")

    # Markdown snippet for your write-up
    md = []
    md.append("# U-CoT Summary (auto-generated)\n")
    for r in rows:
        md.append(f"**{r['group'].title()}** (n={r['n']}): "
                  f"acc={r['accuracy']:.3f}, "
                  f"hint_follow={r['hint_follow']:.3f}, "
                  f"deny_but_use={r['deny_but_use']:.3f}, "
                  f"admit_but_ignore={r['admit_but_ignore']:.3f}, "
                  f"probe_acc={r['probe_acc']:.3f}")
    (out / "ucot_summary.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print("Wrote:", out / "ucot_summary.md")

if __name__ == "__main__":
    main()
