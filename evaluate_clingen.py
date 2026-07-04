# =============================================================================
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python evaluate_clingen.py \
#       --checkpoint outputs/multiseed/stage4/seed_42/best_model_trainable \
#       --data_csv   datasets/ACI-TaskB-TestingSet.csv \
#       --out_dir    outputs/eval/seed_42 \
#       --batch_size 1

# =============================================================================

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import evaluate

import clingen_common as C


# -----------------------------------------------------------------------------
# Metric backends — loaded lazily, identical config to Stage4.py.
# -----------------------------------------------------------------------------
class Metrics:
    def __init__(self, enable_bertscore=True, enable_bleurt=True, enable_unieval=True):
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.sacrebleu = evaluate.load("sacrebleu")
        self.enable_bertscore = enable_bertscore
        self.enable_bleurt = enable_bleurt
        self.enable_unieval = enable_unieval
        self.bertscore = None
        self.bleurt = None
        self.unieval = None

    def _lazy(self):
        if self.enable_bertscore and self.bertscore is None:
            self.bertscore = evaluate.load("bertscore")
        if self.enable_bleurt and self.bleurt is None:
            try:
                self.bleurt = evaluate.load("bleurt", config_name="bleurt-20")
            except Exception as e:
                print(f"[Metric Warning] BLEURT unavailable: {e}")
                self.bleurt = False
        if self.enable_unieval and self.unieval is None:
            if C.__dict__.get("get_evaluator") is not None:
                pass  # not imported at module top; handled below
            try:
                from metric.evaluator import get_evaluator  # UniEval on sys.path
                self.unieval = get_evaluator("summarization")
            except Exception as e:
                print(f"[Metric Warning] UniEval unavailable: {e}")
                self.unieval = False

    def per_example_rouge(self, pred: str, ref: str) -> Dict[str, float]:
        r = self.rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
        return {
            "rouge1": float(r["rouge1"]),
            "rouge2": float(r["rouge2"]),
            "rougeL": float(r["rougeL"]),
            "rougeLsum": float(r.get("rougeLsum", r["rougeL"])),
        }

    def aggregate(self, preds: List[str], refs: List[str], dialogues: List[str]) -> Dict[str, float]:
        r = self.rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        m = self.meteor.compute(predictions=preds, references=refs)
        b = self.sacrebleu.compute(predictions=preds, references=[[x] for x in refs])
        out = {
            "rouge1": float(r["rouge1"]),
            "rouge2": float(r["rouge2"]),
            "rougeL": float(r["rougeL"]),
            "rougeLsum": float(r.get("rougeLsum", r["rougeL"])),
            "meteor": float(m["meteor"]),
            "bleu": float(b["score"]),
        }
        self._lazy()
        if self.bertscore not in (None, False):
            try:
                bs = self.bertscore.compute(
                    predictions=preds, references=refs, lang="en",
                    device="cuda" if torch.cuda.is_available() else "cpu", batch_size=1,
                )
                out["bertscore_f1"] = float(np.mean(bs["f1"]))
            except Exception as e:
                print(f"[Metric Warning] BERTScore failed: {e}")
        if self.bleurt not in (None, False):
            try:
                bl = self.bleurt.compute(predictions=preds, references=refs)
                out["bleurt"] = float(np.mean(bl["scores"]))
            except Exception as e:
                print(f"[Metric Warning] BLEURT failed: {e}")
        if self.unieval not in (None, False):
            try:
                from utils import convert_to_json
                data = convert_to_json(output_list=preds, src_list=dialogues, ref_list=refs)
                scores = self.unieval.evaluate(data, print_result=False)
                if isinstance(scores, list) and scores and isinstance(scores[0], dict):
                    for k in scores[0].keys():
                        out[f"unieval_{k}"] = float(np.mean([d[k] for d in scores]))
                elif isinstance(scores, dict):
                    for k, v in scores.items():
                        out[f"unieval_{k}"] = float(v)
            except Exception as e:
                print(f"[Metric Warning] UniEval failed: {e}")
        return out


# -----------------------------------------------------------------------------
# Generation over a data split.
# -----------------------------------------------------------------------------
def generate_predictions(model, tokenizer, data_list, references, dialogues, batch_size, eval_max_len):
    preds: List[str] = []
    n = len(data_list)
    for start in range(0, n, batch_size):
        batch = data_list[start:start + batch_size]
        coll = C.collate_eval(batch, eval_max_len=eval_max_len)
        gen_tokens = model.generate_note(
            input_ids=coll["input_ids"],
            attention_mask=coll["attention_mask"],
            prompt_embeds=coll["prompt_embeds"],
            tokenizer=tokenizer,
        )
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for d in decoded:
            preds.append(C.postprocess_generated_note(d))
        print(f"[Gen] {min(start + batch_size, n)}/{n}")
        C.cleanup_memory()
    # Canonicalize references the same way training does.
    refs = [C.canonicalize_note(str(r).strip()) for r in references]
    return preds, refs


def evaluate_checkpoint(checkpoint, data_csv, out_dir, batch_size, eval_max_len,
                        enable_bertscore, enable_bleurt, enable_unieval, seed=None):
    os.makedirs(out_dir, exist_ok=True)
    if seed is not None:
        C.set_all_seeds(seed)

    df = pd.read_csv(data_csv)
    dialogues = df["dialogue"].astype(str).tolist()
    references = df["note"].astype(str).tolist()
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]

    model, tokenizer, enc_h = C.load_trained_model(checkpoint)
    data_list = C.build_examples(dialogues, references, tokenizer, enc_h)

    preds, refs = generate_predictions(
        model, tokenizer, data_list, references, dialogues, batch_size, eval_max_len
    )

    # Per-example dump.
    metrics = Metrics(enable_bertscore, enable_bleurt, enable_unieval)
    per_example: List[Dict[str, Any]] = []
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for i, (pid, dlg, ref, pred) in enumerate(zip(ids, dialogues, refs, preds)):
            rec = {"id": pid, "dialogue": dlg, "reference": ref, "prediction": pred}
            f.write(json.dumps(rec) + "\n")
            pe = metrics.per_example_rouge(pred, ref)
            per_example.append({"id": pid, **pe})

    agg = metrics.aggregate(preds, refs, dialogues)

    raw = {
        "checkpoint": checkpoint,
        "data_csv": data_csv,
        "seed": seed,
        "num_examples": len(preds),
        "config": {
            "model": C.MODEL_NAME,
            "prompt_len": C.PROMPT_LEN,
            "soft_prompt_len": C.SOFT_PROMPT_LEN,
            "ner_label_count": C.NER_LABEL_COUNT,
            "lora_r": C.LORA_R, "lora_alpha": C.LORA_ALPHA,
            "lora_target_modules": C.LORA_TARGET_MODULES,
            "gen": {
                "max_new_tokens": C.GEN_MAX_NEW_TOKENS, "min_new_tokens": C.GEN_MIN_NEW_TOKENS,
                "num_beams": C.GEN_NUM_BEAMS, "do_sample": C.GEN_DO_SAMPLE,
                "repetition_penalty": C.GEN_REPETITION_PENALTY, "no_repeat_ngram_size": C.GEN_NO_REPEAT_NGRAM,
            },
        },
        "aggregate": agg,
        "per_example": per_example,
    }
    with open(os.path.join(out_dir, "results_raw.json"), "w") as f:
        json.dump(raw, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[Eval done] {checkpoint}")
    for k, v in agg.items():
        print(f"  {k:<18} {v:.4f}")
    print(f"[Saved] {out_dir}/results_raw.json  metrics.json  predictions.jsonl")
    del model
    C.cleanup_memory()
    return agg


def summarize_seeds(all_metrics: Dict[str, Dict[str, float]], out_dir: str):
    """Mean +/- std across seeds -> multiseed_eval_summary.json (matches Table 8)."""
    keys = sorted({k for m in all_metrics.values() for k in m.keys()})
    summary = {}
    print("\n" + "=" * 72)
    print(f"{'Metric':<20}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
    print("-" * 72)
    for k in keys:
        vals = np.array([float(m[k]) for m in all_metrics.values() if k in m], dtype=float)
        if vals.size == 0:
            continue
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                      "min": float(vals.min()), "max": float(vals.max()),
                      "values": {s: all_metrics[s].get(k) for s in all_metrics}}
        print(f"{k:<20}{vals.mean():>10.4f}{vals.std():>10.4f}{vals.min():>10.4f}{vals.max():>10.4f}")
    with open(os.path.join(out_dir, "multiseed_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out_dir}/multiseed_eval_summary.json")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a ClinGen-DSDA checkpoint on a labeled split.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a *_trainable checkpoint dir (contains lora_adapter/ + custom_trainable_modules.pt).")
    p.add_argument("--checkpoint_glob", type=str, default=None,
                   help="Glob over multiple seed checkpoints for a mean+/-std summary.")
    p.add_argument("--data_csv", type=str, required=True, help="CSV with columns: dialogue,note[,id].")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_max_len", type=int, default=int(os.environ.get("EVAL_MAX_LEN", "1536")))
    p.add_argument("--no_bertscore", action="store_true")
    p.add_argument("--no_bleurt", action="store_true")
    p.add_argument("--no_unieval", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    enable_bs = not args.no_bertscore
    enable_bl = not args.no_bleurt
    enable_ue = not args.no_unieval

    if args.checkpoint_glob:
        ckpts = sorted(glob.glob(args.checkpoint_glob))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob}")
        all_metrics: Dict[str, Dict[str, float]] = {}
        for ck in ckpts:
            # Try to recover a seed id from the path for reproducibility.
            seed = None
            for part in ck.split(os.sep):
                if part.startswith("seed_"):
                    try:
                        seed = int(part.split("_")[1])
                    except Exception:
                        seed = None
            tag = f"seed_{seed}" if seed is not None else os.path.basename(os.path.dirname(ck))
            sub_out = os.path.join(args.out_dir, tag)
            agg = evaluate_checkpoint(ck, args.data_csv, sub_out, args.batch_size, args.eval_max_len,
                                      enable_bs, enable_bl, enable_ue, seed=seed)
            all_metrics[tag] = agg
        summarize_seeds(all_metrics, args.out_dir)
    else:
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint or --checkpoint_glob.")
        evaluate_checkpoint(args.checkpoint, args.data_csv, args.out_dir, args.batch_size, args.eval_max_len,
                            enable_bs, enable_bl, enable_ue, seed=None)


if __name__ == "__main__":
    main()
