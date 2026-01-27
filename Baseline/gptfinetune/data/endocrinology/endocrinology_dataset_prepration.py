import json
import pandas as pd
from datasets import load_dataset
import os

# ============================================================
# Create "system" prompt used for all samples
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert medical professional. Given a clinical dialogue, create a clinical note "
    "outlining key dialogue aspects such as 'CHIEF COMPLAINT' (or 'CC'), 'HISTORY OF PRESENT "
    "ILLNESS' ('HPI'), 'REVIEW OF SYSTEMS', 'PHYSICAL EXAMINATION', 'VITALS REVIEWED', 'RESULTS', "
    "'ASSESSMENT AND PLAN', 'INSTRUCTIONS', 'CURRENT MEDICATIONS', 'PAST MEDICAL HISTORY', 'EXAM', "
    "'IMPRESSION', 'PLAN', 'ASSESSMENT', 'PAST HISTORY', 'ALLERGIES', 'SOCIAL HISTORY', 'PHYSICAL EXAM', "
    "'PROCEDURE', 'FAMILY HISTORY', 'MEDICATIONS', 'VITALS', 'MEDICAL HISTORY', 'SURGICAL HISTORY'. "
    "You will not use all of these aspects in every dialogue, vary it from dialogue to dialogue."
)

# ============================================================
# Endocrinology Dataset Loader
# ============================================================
def load_endocrinology():
    print("🔹 Loading Endocrinology datasets from HuggingFace...")

    train_data = load_dataset("starfishdata/endocrinology_structured_notes_1500")["train"]
    val_data   = load_dataset("starfishdata/endocrinology_structured_notes_eval")["train"]
    test_data  = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]

    # Convert to Pandas DataFrames
    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)

    required_cols = {"transcript", "structured_note"}
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} set missing required columns: {missing}")

    return train_df, val_df, test_df


# ============================================================
# Convert into ChatML Format
# ============================================================
def convert_to_chat_format(df, split):
    """Convert endocrinology dataset into the same message format used in your GPT fine-tuning."""
    records = []

    for idx, row in df.iterrows():
        dialogue = str(row["transcript"]).strip()
        note = str(row["structured_note"]).strip()

        # Build ChatML message structure
        record = {
            "id": f"{split}_{idx}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Dialogue:{dialogue}"},
                {"role": "assistant", "content": note}
            ]
        }

        records.append(record)

    return records


# ============================================================
# Save JSONL File
# ============================================================
def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Saved: {path} ({len(records)} samples)")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main(output_dir="/home/raza/EMRV1/EMRV2/Baseline/gptfinetune/data/endocrinology"):
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = load_endocrinology()

    print("🔹 Converting to ChatML format...")

    train_records = convert_to_chat_format(train_df, "train")
    val_records   = convert_to_chat_format(val_df, "val")
    test_records  = convert_to_chat_format(test_df, "test")

    print("🔹 Saving JSONL files...")
    save_jsonl(train_records, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_records, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_records, os.path.join(output_dir, "test.jsonl"))

    print("🎉 Endocrinology dataset preparation complete!")


if __name__ == "__main__":
    main()
