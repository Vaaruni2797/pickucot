# plots_ucot_causal.py
# Plots for causal hint-use evaluation (CHE / HTR / Δlogprob) + confusion matrix on causal subset.
# Uses only matplotlib (no seaborn). Writes PNGs + CSV + a short markdown.

import json, argparse, csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_json(p: Path):
    J = json.loads(p.read_text(encoding="utf-8"))
    return J.get("summary", {}), J.get("records", [])

def mean_safe(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return float(np.mean(xs)) if xs else float("nan")

def build_per_type(records):
    out = {}
    for t in ["helpful","misleading","irrelevant"]:
        R = [r for r in records if r.get("hint_type")==t]
        if not R:
            out[t] = {"n":0}
            continue
        out[t] = {
            "n": len(R),
            "acc_nohint":       mean_safe([r.get("acc_nohint") for r in R]),
            "acc_with":         mean_safe([r.get("acc_with") for r in R]),
            "acc_flip":         mean_safe([r.get("acc_flip") for r in R]),
            "CHE_acc_mean":     mean_safe([r.get("CHE_acc") for r in R]),
            "HTR_mean":         mean_safe([r.get("HTR") for r in R]),
            "hint_follow_with": mean_safe([r.get("hint_follow_with") for r in R]),
            "hint_follow_flip": mean_safe([r.get("hint_follow_flip") for r in R]),
            "delta_logprob":    mean_safe([r.get("delta_logprob_hint") for r in R]),
        }
    return out

def is_causal(r):
    # Causal if numeric hint exists AND (hint improved accuracy OR answer tracked flipped hint)
    hv = r.get("hint_value")
    che = r.get("CHE_acc", None)
    htr = r.get("HTR", None)
    che_pos = (che is not None) and (float(che) > 0.0)
    htr_one = (htr is not None) and (float(htr) >= 1.0 - 1e-9)
    return (hv is not None) and (che_pos or htr_one)

def bar(ax, labels, values, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6)
    for xi, yi in zip(x, values):
        if isinstance(yi, float) and not (math.isnan(yi) or math.isinf(yi)):
            ax.text(xi, yi + (0.01 if "acc" in ylabel.lower() or "rate" in ylabel.lower() else 0.5),
                    f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

def grouped_bars(ax, groups, series_dict, title, ylabel):
    # series_dict: name -> list of values (same length as groups)
    x = np.arange(len(groups))
    names = list(series_dict.keys())
    n = len(names)
    width = 0.35 if n==2 else 0.22
    offsets = (np.arange(n) - (n-1)/2) * (width + 0.04)
    for i, name in enumerate(names):
        vals = series_dict[name]
        ax.bar(x + offsets[i], vals, width, label=name)
        for xi, yi in zip(x + offsets[i], vals):
            if isinstance(yi, float) and not (math.isnan(yi) or math.isinf(yi)):
                ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

def confusion_on_causal(records):
    C = {"used_yes->copied_yes":0, "used_yes->copied_no":0, "used_no->copied_yes":0, "used_no->copied_no":0}
    causal = [r for r in records if is_causal(r)]
    for r in causal:
        used = r.get("with_used")
        copied = bool(r.get("hint_follow_with")) if r.get("hint_follow_with") is not None else False
        uy = (used == "yes")
        cy = copied
        if uy and cy:   C["used_yes->copied_yes"] += 1
        if uy and not cy: C["used_yes->copied_no"] += 1
        if (not uy) and cy: C["used_no->copied_yes"] += 1
        if (not uy) and (not cy): C["used_no->copied_no"] += 1
    return C, len(causal)

def plot_confusion(ax, C):
    # 2x2: rows = USED_HINT (yes,no), cols = copied (yes,no)
    mat = np.array([
        [C["used_yes->copied_yes"], C["used_yes->copied_no"]],
        [C["used_no->copied_yes"],  C["used_no->copied_no"]],
    ], dtype=float)
    total = mat.sum()
    ax.imshow(mat, aspect="equal")
    for i in range(2):
        for j in range(2):
            val = int(mat[i,j])
            pct = (val/total*100.0) if total>0 else 0.0
            ax.text(j, i, f"{val}\n{pct:.1f}%", ha="center", va="center", fontsize=11, color="white")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["copied: yes", "copied: no"])
    ax.set_yticklabels(["USED_HINT: yes", "USED_HINT: no"])
    ax.set_title("Causal subset — USED_HINT vs actually copied")
    ax.grid(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="causal_eval.json (with 'records' saved)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    p = Path(args.in_json)
    out = Path(args.out_dir) if args.out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    summary, records = load_json(p)
    if not records:
        raise ValueError("No 'records' found. Re-run ucot_causal_eval.py with full records.")

    # ---------- Overall accuracy plot ----------
    labels = ["no-hint", "with-hint", "flipped-hint"]
    acc_vals = [
        float(summary.get("acc_nohint", float("nan"))),
        float(summary.get("acc_with", float("nan"))),
        float(summary.get("acc_flip", float("nan"))),
    ]
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    bar(ax, labels, acc_vals, title="Answer accuracy by condition", ylabel="accuracy")
    plt.tight_layout()
    f1 = out / "overall_acc.png"
    plt.savefig(f1, dpi=220); plt.close()
    print("Wrote:", f1)

    # ---------- Per-type aggregates ----------
    per_type = build_per_type(records)
    types = ["helpful","misleading"]  # irrelevant has no numeric hint
    che_vals = [per_type[t].get("CHE_acc_mean", float("nan")) for t in types]
    htr_vals = [per_type[t].get("HTR_mean", float("nan")) for t in types]
    dlp_vals = [per_type[t].get("delta_logprob", float("nan")) for t in types]

    # CHE & HTR grouped bars
    plt.figure(figsize=(7,4))
    ax = plt.gca()
    grouped_bars(ax, [t.title() for t in types],
                 {"CHE (Δacc)": che_vals, "HTR (flip→follow)": htr_vals},
                 title="Causal influence by hint type",
                 ylabel="rate (CHE, HTR)")
    plt.tight_layout()
    f2 = out / "per_type_che_htr.png"
    plt.savefig(f2, dpi=220); plt.close()
    print("Wrote:", f2)

    # Δ log-prob separate (different scale)
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    bar(ax, [t.title() for t in types], dlp_vals, title="Δ log-prob for hinted digits (with − no-hint)", ylabel="nats")
    plt.tight_layout()
    f3 = out / "per_type_deltalogprob.png"
    plt.savefig(f3, dpi=220); plt.close()
    print("Wrote:", f3)

    # ---------- Confusion matrix on causal subset ----------
    C, n_causal = confusion_on_causal(records)
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    plot_confusion(ax, C)
    plt.tight_layout()
    f4 = out / "causal_confusion.png"
    plt.savefig(f4, dpi=220); plt.close()
    print("Wrote:", f4, f"(n_causal={n_causal})")

    # ---------- CSV + Markdown ----------
    csv_path = out / "ucot_causal_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","overall"])
        for k in ["acc_nohint","acc_with","acc_flip","CHE_acc_mean","HTR_mean","delta_logprob_mean"]:
            w.writerow([k, summary.get(k, "")])
        w.writerow([])
        w.writerow(["type","n","acc_nohint","acc_with","acc_flip","CHE_acc_mean","HTR_mean","hint_follow_with","hint_follow_flip","delta_logprob"])
        for t in ["helpful","misleading","irrelevant"]:
            d = per_type[t]
            w.writerow([t, d.get("n",0), d.get("acc_nohint",""), d.get("acc_with",""), d.get("acc_flip",""),
                        d.get("CHE_acc_mean",""), d.get("HTR_mean",""), d.get("hint_follow_with",""),
                        d.get("hint_follow_flip",""), d.get("delta_logprob","")])
        w.writerow([])
        w.writerow(["causal_confusion_counts"] + list(C.keys()))
        w.writerow(["counts"] + [C[k] for k in C.keys()])
    print("Wrote:", csv_path)

    md = []
    md.append("# Causal U-CoT — summary\n")
    md.append(f"- **Accuracy**: no-hint={summary.get('acc_nohint'):.3f}, with-hint={summary.get('acc_with'):.3f}, flipped={summary.get('acc_flip'):.3f}\n")
    md.append(f"- **CHE (Δacc)** overall: {summary.get('CHE_acc_mean'):.3f}\n")
    md.append(f"- **HTR** overall: {summary.get('HTR_mean'):.3f}\n")
    md.append(f"- **Δ log-prob (hint digits)** overall: {summary.get('delta_logprob_mean'):.3f} nats\n")
    for t in ["helpful","misleading"]:
        d = per_type[t]
        md.append(f"- **{t.title()}**: acc_with={d.get('acc_with'):.3f}, acc_nohint={d.get('acc_nohint'):.3f}, "
                  f"CHE={d.get('CHE_acc_mean'):.3f}, HTR={d.get('HTR_mean'):.3f}, Δlogprob={d.get('delta_logprob'):.2f}")
    md.append(f"\nCausal subset size: **{n_causal}**. See **causal_confusion.png** for USED_HINT vs copied.")
    (out / "ucot_causal_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote:", out / "ucot_causal_summary.md")

if __name__ == "__main__":
    main()
