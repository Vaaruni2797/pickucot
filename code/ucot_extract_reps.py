# step2_ucot_extract_reps.py
# Extract assistant-start hidden representations for 2-turn U-CoT prompts.
# Input : data/ucot/{train,val,test}.jsonl  (from Step 1)
# Output: artifacts/ucot_asst/{split}_reps.npz with:
#   X   : [N, L+1, H]  (all layers incl. embeddings at the assistant-start pos)
#   y   : [N]          (label_causal: 1 if helpful-hint, else 0)
#   ids : [N]          (string ids)
#   meta: dict         (model, hidden_size, etc.)

import os, json, argparse, gc
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_split(jsonl_path: Path) -> Tuple[List[dict], np.ndarray, List[str]]:
    rows, labels, ids = [], [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            labels.append(int(r["label_causal"]))  # 1 if helpful, else 0
            ids.append(r.get("id", ""))
    return rows, np.array(labels, dtype=np.int64), ids

def find_layers_list(model):
    import torch.nn as nn
    for path in ["model.layers","model.model.layers","transformer.h",
                 "model.decoder.layers","gpt_neox.layers","transformer.layers","transformer.blocks"]:
        obj = model; ok = True
        for p in path.split("."):
            if hasattr(obj,p): obj = getattr(obj,p)
            else: ok=False; break
        if ok and hasattr(obj,"__len__") and len(obj)>0: return obj, path
    # fallback: any ModuleList with length >= 2
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module)>=2:
            return module, name
    raise RuntimeError("Could not locate layers list for this architecture.")

def encode_two_turn(tok, turn1: str, turn2: str, max_length: int = 512):
    """
    Chat-encode: [user: turn1], [user: turn2], then assistant-start.
    Return tensors with add_generation_prompt=True so last position is the assistant tag before turn-2 reply.
    """
    msgs = [{"role":"user","content":turn1},
            {"role":"user","content":turn2}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True
    )  # [1, S]
    attn = torch.ones_like(ids)
    if ids.shape[1] > max_length:
        ids = ids[:, -max_length:]
        attn = attn[:, -max_length:]
    return {"input_ids": ids, "attention_mask": attn}

@torch.inference_mode()
def batch_asststart_reps(rows: List[dict], tok, model, max_length=512, batch_size=16) -> np.ndarray:
    """
    Returns X with shape [N, L+1, H]: all layers (incl. embeddings) at the assistant-start position.
    """
    X_parts = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i+batch_size]

        ids_list = []
        for r in chunk:
            enc = encode_two_turn(tok, r["turn1"], r["turn2"], max_length=max_length)
            ids_list.append(enc["input_ids"])
        # left-pad within batch
        ids = torch.nn.utils.rnn.pad_sequence(
            [x[0] for x in ids_list], batch_first=True, padding_value=pad_id
        )
        attn = (ids != pad_id).long()
        enc = {"input_ids": ids.to(model.device), "attention_mask": attn.to(model.device)}

        out = model(**enc)  # output_hidden_states=True set at model load
        # hidden_states: tuple length L+1, each [B,S,H]
        hs = torch.stack(out.hidden_states)  # [L+1,B,S,H]
        last_idx = enc["attention_mask"].sum(dim=1) - 1  # [B], assistant-start position
        # gather layerwise at last_idx
        sel = torch.stack([hs[:, b, last_idx[b], :] for b in range(hs.shape[1])], dim=1)  # [L+1,B,H]
        # cast to a CPU-friendly dtype (avoid bfloat16 numpy issue)
        sel_np = sel.to(torch.float16 if torch.cuda.is_available() else torch.float32).cpu().numpy()
        X_parts.append(sel_np)

        del out, hs, sel, enc, ids, attn, ids_list; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    X = np.concatenate(X_parts, axis=1)  # [L+1,N,H]
    X = np.swapaxes(X, 0, 1)             # [N,L+1,H]
    return X

def save_npz(path: Path, X, y, ids, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, ids=np.array(ids), meta=np.array([meta], dtype=object))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("MODEL_ID","TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--data_dir", default="data/ucot")
    ap.add_argument("--out_dir",  default="artifacts/ucot_asst")
    ap.add_argument("--dtype",    default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    torch_dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        output_hidden_states=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    layers, _ = find_layers_list(model)
    H = next(p for _,p in model.named_parameters() if p.ndim>=2).shape[-1]
    meta_common = {
        "model": args.model,
        "num_layers_plus_embed": int(len(layers)+1),
        "hidden_size": int(H),
        "pos": "assistant_start(turn2)",
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
    }

    for split in ["train","val","test"]:
        rows, y, ids = load_split(Path(args.data_dir)/f"{split}.jsonl")
        X = batch_asststart_reps(rows, tok, model,
                                 max_length=args.max_length,
                                 batch_size=args.batch_size)
        meta = dict(meta_common, split=split)
        save_npz(Path(args.out_dir)/f"{split}_reps.npz", X, y, ids, meta)
        print(f"Saved {split}: X{tuple(X.shape)} y{tuple(y.shape)} -> {Path(args.out_dir)/f'{split}_reps.npz'}")

if __name__ == "__main__":
    main()