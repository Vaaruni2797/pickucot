# step3_ucot_train_probes.py
# Train per-layer probes for U-CoT (hint causal relevance) on assistant-start reps.
# Input : artifacts/ucot_asst/{train,val,test}_reps.npz  (from Step 2)
# Output: artifacts/ucot_asst/probes/
#   - probe_acc_vs_layer.png
#   - per_layer_metrics.csv
#   - best_probe_w.npy, best_probe_b.npy, best_probe_meta.json
#   - (optional auditor) mlp metrics in CSV

import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)   # [N, L+1, H]
    y = d["y"].astype(np.int64)
    ids = d["ids"]

    meta = {}
    if "meta" in d:
        meta_raw = d["meta"]
        if isinstance(meta_raw, dict):
            meta = meta_raw
        elif isinstance(meta_raw, np.ndarray):
            # saved as np.array([meta], dtype=object)
            if meta_raw.dtype == object and meta_raw.size > 0:
                first = meta_raw.reshape(-1)[0]
                if isinstance(first, dict):
                    meta = first
    return X, y, ids, meta


def train_logreg(Xtr, ytr, Xva, yva, Cs=(0.1, 0.3, 1.0, 3.0, 10.0), class_weight="balanced", standardize=True, seed=7):
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_ = scaler.fit_transform(Xtr)
        Xva_ = scaler.transform(Xva)
    else:
        Xtr_, Xva_ = Xtr, Xva
    best = {"acc": -1, "C": None, "w": None, "b": None, "scaler": scaler}
    for C in Cs:
        clf = LogisticRegression(
            penalty="l2", C=C, class_weight=class_weight,
            solver="liblinear", max_iter=2000, random_state=seed
        )
        clf.fit(Xtr_, ytr)
        preds = clf.predict(Xva_)
        acc = accuracy_score(yva, preds)
        if acc > best["acc"]:
            best.update({
                "acc": acc,
                "C": C,
                "w": clf.coef_.reshape(-1).astype(np.float32),
                "b": float(clf.intercept_.reshape(-1)[0]),
            })
    return best

def eval_probe(w, b, X, y, scaler=None):
    if scaler is not None:
        X_ = scaler.transform(X)
    else:
        X_ = X
    logits = X_ @ w + b
    preds = (logits > 0).astype(np.int64)
    return accuracy_score(y, preds)

def train_mlp(Xtr, ytr, Xva, yva, hidden=128, standardize=True, seed=7):
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_ = scaler.fit_transform(Xtr)
        Xva_ = scaler.transform(Xva)
    else:
        Xtr_, Xva_ = Xtr, Xva
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                        max_iter=300, random_state=seed, early_stopping=True)
    clf.fit(Xtr_, ytr)
    acc = accuracy_score(yva, clf.predict(Xva_))
    return {"acc": acc, "model": clf, "scaler": scaler}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default="artifacts/ucot_asst")
    ap.add_argument("--out_dir", default="artifacts/ucot_asst/probes")
    ap.add_argument("--standardize", action="store_true", help="Standardize features before probing")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, ids_tr, meta_tr = load_npz(in_dir / "train_reps.npz")
    Xva, yva, ids_va, meta_va = load_npz(in_dir / "val_reps.npz")
    Xte, yte, ids_te, meta_te = load_npz(in_dir / "test_reps.npz")

    N, Lp1, H = Xtr.shape
    print(f"Loaded reps: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # Train per-layer logistic probe (and optional MLP auditor)
    per_layer = []
    val_accs = []
    for layer in range(Lp1):  # 0=embeddings, 1..L=blocks
        Xtr_l = Xtr[:, layer, :]
        Xva_l = Xva[:, layer, :]
        Xte_l = Xte[:, layer, :]

        # Logistic
        best = train_logreg(Xtr_l, ytr, Xva_l, yva, standardize=args.standardize, seed=args.seed)
        test_acc = eval_probe(best["w"], best["b"], Xte_l, yte, scaler=best["scaler"])

        # Tiny MLP auditor (optional; keeps numbers separate)
        mlp = train_mlp(Xtr_l, ytr, Xva_l, yva, hidden=128, standardize=args.standardize, seed=args.seed)
        # We'll evaluate MLP on test only for reference
        if args.standardize:
            Xte_m = mlp["scaler"].transform(Xte_l)
        else:
            Xte_m = Xte_l
        mlp_test_acc = accuracy_score(yte, mlp["model"].predict(Xte_m))

        row = {
            "layer": layer,
            "val_acc_logreg": round(best["acc"], 4),
            "test_acc_logreg": round(test_acc, 4),
            "C": best["C"],
            "val_acc_mlp": round(mlp["acc"], 4),
            "test_acc_mlp": round(mlp_test_acc, 4),
        }
        per_layer.append(row)
        val_accs.append(best["acc"])
        print(f"Layer {layer:>2}: logreg val={row['val_acc_logreg']:.3f} test={row['test_acc_logreg']:.3f} | mlp val={row['val_acc_mlp']:.3f} test={row['test_acc_mlp']:.3f}")

        # stash best weights to consider later
        row["_w"] = best["w"]; row["_b"] = best["b"]; row["_scaler"] = best["scaler"]

    # Pick best layer by val (logreg)
    best_idx = int(np.argmax(val_accs))
    best_row = per_layer[best_idx]
    print(f"\nBest layer by val (logreg): {best_idx}  val={best_row['val_acc_logreg']:.3f}  test={best_row['test_acc_logreg']:.3f}")

    # Save plot
    plt.figure()
    plt.title("U-CoT probe accuracy vs layer (assistant-start)")
    plt.plot([r["val_acc_logreg"] for r in per_layer], label="val (logreg)")
    plt.plot([r["test_acc_logreg"] for r in per_layer], label="test (logreg)")
    plt.xlabel("hidden_states layer index (0=embeddings)")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "probe_acc_vs_layer.png", dpi=200)
    plt.close()

    # Save CSV
    with (out_dir / "per_layer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer","val_acc_logreg","test_acc_logreg","C","val_acc_mlp","test_acc_mlp"])
        w.writeheader(); [w.writerow({k:v for k,v in r.items() if not k.startswith("_")}) for r in per_layer]

    # Save best probe weights
    np.save(out_dir / "best_probe_w.npy", best_row["_w"])
    np.save(out_dir / "best_probe_b.npy", np.array([best_row["_b"]], dtype=np.float32))
    # For convenience (even though we’re not doing UTMS), also save a unit-norm vector
    w = best_row["_w"]; w_unit = w / (np.linalg.norm(w) + 1e-8)
    np.save(out_dir / "best_probe_w_unit.npy", w_unit.astype(np.float32))

    # Save meta
    meta = {
        "task": "ucot (hint causal relevance)",
        "pos": "assistant_start(turn2)",
        "best_layer": best_idx,
        "val_acc_logreg": best_row["val_acc_logreg"],
        "test_acc_logreg": best_row["test_acc_logreg"],
        "val_acc_mlp": best_row["val_acc_mlp"],
        "test_acc_mlp": best_row["test_acc_mlp"],
        "standardize": bool(args.standardize),
        "seed": int(args.seed)
    }
    (out_dir / "best_probe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved best probe + meta to {out_dir}")
    print(f"Plot: {out_dir / 'probe_acc_vs_layer.png'}")
    print(f"CSV : {out_dir / 'per_layer_metrics.csv'}")

if __name__ == "__main__":
    main()