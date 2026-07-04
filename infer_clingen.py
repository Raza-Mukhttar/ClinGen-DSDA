# =============================================================================

#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python infer_clingen.py \
#       --checkpoint outputs/multiseed/stage4/seed_42/best_model_trainable \
#       --input_csv  datasets/new_dialogues.csv \
#       --out_jsonl  outputs/inference/notes.jsonl \
#       --batch_size 1
# =============================================================================

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import clingen_common as C


def read_inputs(args) -> Tuple[List[str], List[str]]:
    """Return (ids, dialogues) from whichever input flag was provided."""
    if args.text is not None:
        return ["0"], [args.text]
    if args.input_txt is not None:
        with open(args.input_txt, "r") as f:
            return ["0"], [f.read()]
    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)
        if "dialogue" not in df.columns:
            raise ValueError("--input_csv must have a 'dialogue' column.")
        dialogues = df["dialogue"].astype(str).tolist()
        ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
        return ids, dialogues
    if args.input_jsonl is not None:
        ids, dialogues = [], []
        with open(args.input_jsonl, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                dialogues.append(str(obj["dialogue"]))
                ids.append(str(obj.get("id", i)))
        return ids, dialogues
    raise ValueError("Provide one of --text / --input_txt / --input_csv / --input_jsonl.")


def run_inference(checkpoint, ids, dialogues, batch_size, eval_max_len) -> List[Dict[str, Any]]:
    model, tokenizer, enc_h = C.load_trained_model(checkpoint)
    # No references at inference time -> pass notes=None (pipeline uses empty targets).
    data_list = C.build_examples(dialogues, None, tokenizer, enc_h)

    results: List[Dict[str, Any]] = []
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
        for j, d in enumerate(decoded):
            idx = start + j
            note = C.postprocess_generated_note(d)
            results.append({"id": ids[idx], "dialogue": dialogues[idx], "note": note})
        print(f"[Infer] {min(start + batch_size, n)}/{n}")
        C.cleanup_memory()

    del model
    C.cleanup_memory()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Generate clinical notes from dialogues with ClinGen-DSDA.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a *_trainable checkpoint dir (lora_adapter/ + custom_trainable_modules.pt).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, default=None, help="A single dialogue string.")
    src.add_argument("--input_txt", type=str, default=None, help="Text file with one dialogue.")
    src.add_argument("--input_csv", type=str, default=None, help="CSV with a 'dialogue' column [, 'id'].")
    src.add_argument("--input_jsonl", type=str, default=None, help="JSONL with {'dialogue','id'} per line.")
    p.add_argument("--out_jsonl", type=str, default=None, help="Where to write {id,dialogue,note} lines.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_max_len", type=int, default=int(os.environ.get("EVAL_MAX_LEN", "1536")))
    return p.parse_args()


def main():
    args = parse_args()
    ids, dialogues = read_inputs(args)
    print(f"[Infer] {len(dialogues)} dialogue(s) from input.")

    results = run_inference(args.checkpoint, ids, dialogues, args.batch_size, args.eval_max_len)

    if args.out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[Saved] {args.out_jsonl}")

    # Single-input convenience: print the note.
    if len(results) == 1:
        print("\n" + "=" * 60 + "\nGENERATED NOTE\n" + "=" * 60)
        print(results[0]["note"])


if __name__ == "__main__":
    main()
