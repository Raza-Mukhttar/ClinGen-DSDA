# for endocrinology dataset
# File: medinotes_repro/scripts/preprocess.py
"""
Preprocess ACI-BENCH + Endocrinology dataset according to paper methodology:
- Data Cleaning
- Text Normalization (medical terms)
- Whisper/Endocrinology-specific noise cleaning
- Instruction Tuning Example Generation (SOAP)
- Optional SentencePiece training

Usage:
    python /Baseline/medinotes/preprocess.py \
        --in /Baseline/medinotes/data/endo_processed.jsonl \
        --out /Baseline/medinotes/endo_instruction_tune.jsonl \
        --spm /Baseline/medinotes/model/endo_spm_medical.model \
        --train_spm
"""

import argparse
import json
import os
import re
import sentencepiece as spm
from tqdm import tqdm


# ----------------------------------------------------------
# 1. Medical Term Normalization Dictionary (expandable)
# ----------------------------------------------------------
MEDICAL_NORMALIZATION = {
    "bp": "blood pressure",
    "hr": "heart rate",
    "sob": "shortness of breath",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "hpi": "history of present illness",
    "cc": "chief complaint",
    "f/u": "follow up",
    "fu": "follow up",
    "pt": "patient",
}


def normalize_medical_terms(text: str) -> str:
    for abbr, full in MEDICAL_NORMALIZATION.items():
        text = re.sub(rf"\b{abbr}\b", full, text, flags=re.IGNORECASE)
    return text


# ----------------------------------------------------------
# 2. Endocrinology & Whisper-Specific Cleaning
# ----------------------------------------------------------
def clean_endocrinology_noise(text: str) -> str:
    """
    Endocrinology dataset has Whisper ASR noise, filler words, timestamps.
    This removes:
        - filler words ("uh", "um", "okay", "you know")
        - stutters ("I I I", "the the the")
        - transcription artifacts ("<unk>", "[noise]")
        - timestamp markers ("[00:00:02]", "(laugh)")
    """

    patterns = [
        r"\buh+\b",
        r"\bum+\b",
        r"\bokay+\b",
        r"\byou know\b",
        r"\bmm-hmm\b",
        r"\b(hm+|hmm+)\b",

        # Stutters like "I I I", "the the"
        r"\b(\w+)\s+\1(\s+\1)?\b",

        # Whisper artifacts
        r"<unk>",
        r"\[noise\]",
        r"\[silence\]",
        r"\[laughter\]",
        r"\(laugh\)",
        r"\(breath\)",
        r"\[.*?\d+.*?\]",      # timestamps like [00:00:03]
    ]

    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)

    return text


# ----------------------------------------------------------
# 3. Global Cleaning & Normalization
# ----------------------------------------------------------
def clean_text(text: str) -> str:
    if text is None or str(text).strip() == "":
        return ""

    text = str(text)

    # Remove ASR/Endo noisy tokens BEFORE normalization
    text = clean_endocrinology_noise(text)

    # Remove irrelevant characters (non-linguistic noise)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\-\n\s]", " ", text)

    # Standardize medical terms
    text = normalize_medical_terms(text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------------------------------
# 4. Instruction Template (from paper Fig. 2)
# ----------------------------------------------------------
INSTRUCTION_TEMPLATE = (
    "Convert the following medical dialogue into a structured SOAP note. "
    "Use the following sections exactly:\n"
    "Subjective:\nObjective:\nAssessment:\nPlan:"
)


def build_instructions(inpath: str, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(inpath, "r", encoding="utf-8") as fin, \
         open(outpath, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Processing dataset"):
            j = json.loads(line)

            dialogue = clean_text(j["dialogue"])
            note = clean_text(j["soap"])

            item = {
                "id": j["id"],
                "instruction": INSTRUCTION_TEMPLATE,
                "input": dialogue,
                "output": note,
                "split": j["split"],
            }

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✔ Instruction tuning dataset saved → {outpath}")


# ----------------------------------------------------------
# 5. SentencePiece Training
# ----------------------------------------------------------
def train_sentencepiece(corpus_path: str, model_prefix: str, vocab_size: int = 16000):
    tmp = "tmp_spm_training.txt"
    with open(tmp, "w", encoding="utf-8") as out:
        with open(corpus_path, "r", encoding="utf-8") as fin:
            for line in fin:
                j = json.loads(line)
                out.write(j["input"] + "\n")
                out.write(j["output"] + "\n")

    spm.SentencePieceTrainer.Train(
        f"--input={tmp} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage=1.0 "
        f"--model_type=bpe"
    )

    os.remove(tmp)
    print(f"✔ SentencePiece model saved → {model_prefix}.model")


# ----------------------------------------------------------
# 6. Main Handler
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--spm", help="SentencePiece model output path")
    parser.add_argument("--train_spm", action="store_true")
    args = parser.parse_args()

    build_instructions(args.infile, args.out)

    if args.train_spm:
        if not args.spm:
            raise ValueError("--spm must be provided when using --train_spm")
        model_prefix = args.spm.rsplit(".", 1)[0]
        train_sentencepiece(args.out, model_prefix)


if __name__ == "__main__":
    main()
