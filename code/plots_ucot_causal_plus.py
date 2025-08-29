# plots_ucot_causal_plus.py
# More explanatory visuals for U-CoT causal eval.
# Inputs: causal_eval.json with full "records".
# Outputs: five PNGs + one CSV in --out_dir (defaults to the JSON's folder).
#
# Figures:
# 1) delta_acc_hist.png      : histogram of CHE per item (-1, 0, +1) with counts & percents
# 2) uplift_curve_htr.png    : HTR vs quantiles of Δ log-prob (hinted digits), helpful & misleading
# 3) flip_outcome_mix.png    : composition (follow flip / stay GT / other) by hint type (helpful, misleading)
# 4) causal_truthfulness.png : deny-but-use & admit-but-ignore rates on the causal subset, overall & by type
# 5) per_type_summary_table.csv : numeric backing for the above

import json, argparse, math, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_records(p: Path):
    J = json.loads(p.read_text(encoding="utf-8"))
    recs = J.get("records", [])
    if not recs:
        raise ValueError("No 'records' found. Re-run ucot_causal_eval.py with full records.")
    return recs

def safe_bool(x):
    return bool(x) if x is not None else False

def is_numeric_hint(r):
    return r.get("hint_value") is not None

def is_flippable(r):
    return r.get("flipped_value") is not None and r.get("flip_final") is not None

def delta_acc(r):
    a_with = 1 if safe_bool(r.get("acc_with")) else 0
    a_no   = 1 if safe_bool(r.get("acc_nohint")) else 0
    return a_with - a_no  # in {-1,0,+1}

def is_causal(r):
    che = r.get("CHE_acc", None)
    htr = r.get("HTR", None)
    che_pos = (che is not None) and (float(che) > 0.0)
    htr_one = (htr is not None) and (float(htr) >= 1.0 - 1e-9)
    return is_numeric_hint(r) and (che_pos or htr_one)

def pct(x, n):
    return 0.0 if n == 0 else 100.0 * x / n

def plot_delta_acc_hist(records, out_path):
    vals = [delta_acc(r) for r in records]
    counts = { -1: vals.count(-1), 0: vals.count(0), 1: vals.count(1) }
    total = len(vals)

    xs = [-1, 0, 1]
    heights = [counts[x] for x in xs]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.bar([0,1,2], heights, width=0.6)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["worse", "same", "better"])
    ax.set_ylabel("count")
    ax.set_title("Change in accuracy with hints (CHE per item)")
    for i, h in enumerate(heights):
        ax.text(i, h + max(1, 0.01*total), f"{h} ({pct(h,total):.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def bin_by_quantiles(values, k=8):
    # returns (bins, edges): bins is list of lists of indices for each bin
    arr = np.array(values, dtype=float)
    qs = np.linspace(0, 1, k+1)
    edges = np.quantile(arr, qs)
    # ensure strictly increasing edges (handle duplicates)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9
    bins = [[] for _ in range(k)]
    for i, v in enumerate(arr):
        # find bin
        j = np.searchsorted(edges, v, side="right") - 1
        j = min(max(j, 0), k-1)
        bins[j].append(i)
    return bins, edges

def plot_uplift_curve_htr(records, out_path, k=8):
    # Use only items with numeric hint, non-nan delta_logprob_hint, and flippable
    filt = [r for r in records if is_numeric_hint(r) and is_flippable(r)
            and (r.get("delta_logprob_hint") is not None)
            and (not math.isnan(float(r.get("delta_logprob_hint"))))]
    if not filt:
        return
    dlp = [float(r["delta_logprob_hint"]) for r in filt]
    bins, edges = bin_by_quantiles(dlp, k=k)

    def agg_rate(mask_fn):
        rates = []; counts = []
        for b in bins:
            sub = [filt[i] for i in b if mask_fn(filt[i])]
            if not b:
                rates.append(float("nan")); counts.append(0); continue
            if not sub:
                rates.append(0.0); counts.append(len(b)); continue
            # HTR in this subset
            vals = [1.0 if safe_bool(r.get("hint_follow_flip")) else 0.0 for r in sub]
            rates.append(float(np.mean(vals)))
            counts.append(len(sub))
        return rates, counts

    helpful_rates, helpful_n = agg_rate(lambda r: r.get("hint_type")=="helpful")
    misleading_rates, misleading_n = agg_rate(lambda r: r.get("hint_type")=="misleading")

    centers = [(edges[i]+edges[i+1])/2.0 for i in range(len(edges)-1)]

    plt.figure(figsize=(7,4))
    ax = plt.gca()
    # Helpful curve
    ax.plot(centers, helpful_rates, marker="o", label="Helpful")
    # Misleading curve
    ax.plot(centers, misleading_rates, marker="s", label="Misleading")
    for x, y, n in zip(centers, helpful_rates, helpful_n):
        if n>0 and not math.isnan(y):
            ax.text(x, y+0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("Δ log-prob for hinted digits (with − no-hint)")
    ax.set_ylabel("Hint-Tracking Rate (follow flipped number)")
    ax.set_title("Uplift curve: HTR rises with hinted-digit probability")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_flip_outcome_mix(records, out_path):
    # Only flippable items, by hint type helpful/misleading
    types = ["helpful","misleading"]
    cats = ["follow_flip", "stay_gt", "other"]
    fracs = {t: {c:0 for c in cats} for t in types}
    counts = {t:0 for t in types}

    for r in records:
        if not is_flippable(r): continue
        t = r.get("hint_type")
        if t not in types: continue
        counts[t] += 1
        ff = (r.get("flip_final") == r.get("flipped_value"))
        sg = (r.get("flip_final") == r.get("gt"))
        if ff: fracs[t]["follow_flip"] += 1
        elif sg: fracs[t]["stay_gt"] += 1
        else: fracs[t]["other"] += 1

    xs = np.arange(len(types))
    data = { c: [fracs[t][c]/counts[t] if counts[t]>0 else 0.0 for t in types] for c in cats }
    bottoms = np.zeros(len(types))

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    for c in cats:
        vals = data[c]
        ax.bar(xs, vals, bottom=bottoms, width=0.6, label=c.replace("_"," "))
        bottoms += np.array(vals)
    ax.set_xticks(xs); ax.set_xticklabels([t.title() for t in types])
    ax.set_ylabel("fraction")
    ax.set_title("Flipped-hint outcomes by hint type")
    for i,t in enumerate(types):
        ax.text(i, 1.02, f"n={counts[t]}", ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_causal_truthfulness(records, out_path):
    # Causal subset: CHE>0 or HTR==1
    causal = [r for r in records if is_causal(r)]
    if not causal:
        return
    groups = ["overall","helpful","misleading"]
    deny = []
    admit = []
    Ns = []
    for g in groups:
        if g=="overall":
            sub = causal
        else:
            sub = [r for r in causal if r.get("hint_type")==g]
        Ns.append(len(sub))
        if not sub:
            deny.append(0.0); admit.append(0.0); continue
        d = [1.0 if safe_bool(r.get("deny_but_use_with")) else 0.0 for r in sub]
        a = [1.0 if safe_bool(r.get("admit_but_ignore_with")) else 0.0 for r in sub]
        deny.append(float(np.mean(d)))
        admit.append(float(np.mean(a)))

    # grouped bars (two series)
    x = np.arange(len(groups))
    width = 0.35
    plt.figure(figsize=(7,4))
    ax = plt.gca()
    ax.bar(x - width/2, deny, width, label="deny-but-use")
    ax.bar(x + width/2, admit, width, label="admit-but-ignore")
    for xi, (d,a,n) in enumerate(zip(deny, admit, Ns)):
        ax.text(xi - width/2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(xi + width/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi, 1.03, f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([g.title() for g in groups])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("rate")
    ax.set_title("Truthfulness on the causal subset")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def write_per_type_csv(records, csv_path):
    types = ["helpful","misleading","irrelevant"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type","n",
                    "acc_with","acc_nohint","acc_flip",
                    "delta_acc_mean",
                    "hint_follow_with","hint_follow_flip",
                    "HTR","deny_but_use_with","admit_but_ignore_with"])
        for t in types:
            sub = [r for r in records if r.get("hint_type")==t]
            n = len(sub)
            def m(key): 
                vv = [r.get(key) for r in sub if r.get(key) is not None and not (isinstance(r.get(key), float) and math.isnan(r.get(key)))]
                return float(np.mean(vv)) if vv else float("nan")
            da = [delta_acc(r) for r in sub]
            w.writerow([t, n,
                        m("acc_with"), m("acc_nohint"), m("acc_flip"),
                        float(np.mean(da)) if da else float("nan"),
                        m("hint_follow_with"), m("hint_follow_flip"),
                        m("HTR"), m("deny_but_use_with"), m("admit_but_ignore_with")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="causal_eval.json with 'records'")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--uplift_bins", type=int, default=8)
    args = ap.parse_args()

    p = Path(args.in_json)
    out = Path(args.out_dir) if args.out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(p)

    plot_delta_acc_hist(records, out / "delta_acc_hist.png")
    plot_uplift_curve_htr(records, out / "uplift_curve_htr.png", k=args.uplift_bins)
    plot_flip_outcome_mix(records, out / "flip_outcome_mix.png")
    plot_causal_truthfulness(records, out / "causal_truthfulness.png")
    write_per_type_csv(records, out / "per_type_summary_table.csv")

    print("Wrote:",
          out / "delta_acc_hist.png", "\n",
          out / "uplift_curve_htr.png", "\n",
          out / "flip_outcome_mix.png", "\n",
          out / "causal_truthfulness.png", "\n",
          out / "per_type_summary_table.csv")

if __name__ == "__main__":
    main()
