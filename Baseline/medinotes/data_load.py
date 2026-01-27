# # File: medinotes_repro/scripts/data_load.py
# """
# Load pre-split ACI datasets (train/val/test) and convert them into
# a unified JSONL format for downstream preprocessing.

# Expected CSV columns:
#     dialogue, note

# Usage:
#     python /home/raza/EMRV1/EMRV2/Baseline/medinotes/data_load.py \
#         --train_csv /home/raza/EMRV1/datasets/ACI-TaskB-TrainingSet.csv \
#         --val_csv /home/raza/EMRV1/datasets/ACI-TaskB-ValidationSet.csv \
#         --test_csv /home/raza/EMRV1/datasets/ACI-TaskB-TestingSet.csv \
#         --out /home/raza/EMRV1/EMRV2/Baseline/medinotes/data/processed.jsonl
# """

# import argparse
# import json
# import os
# import pandas as pd


# def load_csv(path: str, split: str):
#     """Load a CSV and return list of dicts with required fields."""
#     df = pd.read_csv(path)

#     required = {"dialogue", "note"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"{path} missing required columns: {missing}")

#     records = []
#     for idx, row in df.iterrows():
#         records.append({
#             "id": f"{split}_{idx}",
#             "dialogue": str(row["dialogue"]),
#             "soap": str(row["note"]),
#             "split": split
#         })
#     return records


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_csv", required=True)
#     parser.add_argument("--val_csv", required=True)
#     parser.add_argument("--test_csv", required=True)
#     parser.add_argument("--out", required=True)
#     args = parser.parse_args()

#     train_records = load_csv(args.train_csv, "train")
#     val_records = load_csv(args.val_csv, "val")
#     test_records = load_csv(args.test_csv, "test")

#     all_records = train_records + val_records + test_records

#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     with open(args.out, "w", encoding="utf-8") as f:
#         for rec in all_records:
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#     print(f"Saved {len(all_records)} total records → {args.out}")
#     print(f"  Train: {len(train_records)}")
#     print(f"  Val:   {len(val_records)}")
#     print(f"  Test:  {len(test_records)}")


# if __name__ == "__main__":
#     main()








# File: medinotes_repro/scripts/data_load.py
"""
Unified data loader for:
  (A) ACI CSV datasets (dialogue, note)
  (B) Endocrinology HuggingFace datasets:
        - starfishdata/endocrinology_structured_notes_1500
        - starfishdata/endocrinology_structured_notes_eval
        - starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper

Outputs a single JSONL formatted as:

{
    "id": "<split>_<index>",
    "dialogue": "...",
    "soap": "...",
    "split": "train|val|test"
}

Usage (ACI mode):
-----------------
python data_load.py \
    --dataset_type aci \
    --train_csv ACI-TaskB-TrainingSet.csv \
    --val_csv   ACI-TaskB-ValidationSet.csv \
    --test_csv  ACI-TaskB-TestingSet.csv \
    --out data/processed.jsonl

Usage (Endocrinology mode):
---------------------------
python /home/raza/EMRV1/EMRV2/Baseline/medinotes/data_load.py \
    --dataset_type endo \
    --out /home/raza/EMRV1/EMRV2/Baseline/medinotes/data/endo_processed.jsonl
"""

import argparse
import json
import os
import pandas as pd
from datasets import load_dataset


# ============================================================
# ACI CSV Loader
# ============================================================
def load_csv(path: str, split: str):
    df = pd.read_csv(path)

    required = {"dialogue", "note"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": f"{split}_{idx}",
            "dialogue": str(row["dialogue"]),
            "soap": str(row["note"]),
            "split": split
        })
    return records


# ============================================================
# Endocrinology Dataset Loader
# ============================================================
def load_endocrinology():
    print("🔹 Loading Endocrinology datasets from HuggingFace...")

    train_data = load_dataset("starfishdata/endocrinology_structured_notes_1500")["train"]
    val_data   = load_dataset("starfishdata/endocrinology_structured_notes_eval")["train"]
    test_data  = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]

    # Convert into Pandas DataFrames
    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)

    # Required columns: transcript, structured_note
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if "transcript" not in df.columns or "structured_note" not in df.columns:
            raise ValueError(f"{name} set missing required columns: transcript, structured_note")

    train_records = []
    for idx, row in train_df.iterrows():
        train_records.append({
            "id": f"train_{idx}",
            "dialogue": str(row["transcript"]),
            "soap": str(row["structured_note"]),
            "split": "train"
        })

    val_records = []
    for idx, row in val_df.iterrows():
        val_records.append({
            "id": f"val_{idx}",
            "dialogue": str(row["transcript"]),
            "soap": str(row["structured_note"]),
            "split": "val"
        })

    test_records = []
    for idx, row in test_df.iterrows():
        test_records.append({
            "id": f"test_{idx}",
            "dialogue": str(row["transcript"]),
            "soap": str(row["structured_note"]),
            "split": "test"
        })

    return train_records, val_records, test_records


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["aci", "endo"],
        required=True,
        help="Which dataset to preprocess (aci or endo)"
    )

    parser.add_argument("--train_csv", type=str)
    parser.add_argument("--val_csv", type=str)
    parser.add_argument("--test_csv", type=str)

    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # -------------------------
    # ACI mode
    # -------------------------
    if args.dataset_type == "aci":
        if not (args.train_csv and args.val_csv and args.test_csv):
            raise ValueError("ACI mode requires --train_csv --val_csv --test_csv")

        train_records = load_csv(args.train_csv, "train")
        val_records   = load_csv(args.val_csv, "val")
        test_records  = load_csv(args.test_csv, "test")

    # -------------------------
    # Endocrinology mode
    # -------------------------
    elif args.dataset_type == "endo":
        train_records, val_records, test_records = load_endocrinology()

    # -------------------------
    # Write output JSONL
    # -------------------------
    all_records = train_records + val_records + test_records

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(all_records)} total records → {args.out}")
    print(f"  Train: {len(train_records)}")
    print(f"  Val:   {len(val_records)}")
    print(f"  Test:  {len(test_records)}")


if __name__ == "__main__":
    main()
