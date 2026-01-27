
"""
RAG Retriever for MediNotes (Paper-Aligned, FAISS Version)
Supports ACI + Endocrinology datasets.

Features:
- Whisper/Endocrinology cleaning
- Chunking (LangChain)
- SentenceTransformer embeddings
- FAISS vector store
- Dataset-aware metadata

Usage:
    python /Baseline/medinotes/retriever_index.py \
        --in /Baseline/medinotes/endo_instruction_tune.jsonl \
        --out_index /Baseline/medinotes/data/endo_faiss_index \
        --chunk_size 512 \
        --chunk_overlap 64 \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import os
import re
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------------------------------------
# Endocrinology ASR/Whisper noise removal
# ----------------------------------------------------------
def clean_endo_noise(text: str) -> str:

    if text is None:
        return ""

    patterns = [
        r"\buh+\b", r"\bum+\b", r"\bokay+\b",
        r"\byou know\b", r"\bmm-hmm\b",
        r"<unk>", r"\[noise\]", r"\[silence\]", r"\[laughter\]",
        r"\(laugh\)", r"\(breath\)",

        # timestamps
        r"\[\d{1,2}:\d{2}:\d{2}\]",

        # repeated stutter: "I I I" or "the the"
        r"\b(\w+)\s+\1(\s+\1)?\b",
    ]

    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------
# Load instruction-tuning dataset
# ----------------------------------------------------------
def load_examples(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # Determine dataset source automatically
            if "endo" in item["id"].lower():
                item["dataset"] = "endocrinology"
                item["input"] = clean_endo_noise(item["input"])
                item["output"] = clean_endo_noise(item["output"])
            else:
                item["dataset"] = "aci"

            data.append(item)
    return data


# ----------------------------------------------------------
# Build chunks from dialogue + SOAP
# ----------------------------------------------------------
def build_chunks(data, chunk_size, chunk_overlap):

    # Medical-aware splitters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", "Subjective:", "Objective:", "Assessment:", "Plan:",
            "\n", ".", "?", "!", " ", ""
        ]
    )

    docs = []

    for item in data:

        # Dialogue chunks
        d_chunks = splitter.split_text(item["input"])
        for ch in d_chunks:
            docs.append({
                "text": ch,
                "metadata": {
                    "id": item["id"],
                    "dataset": item["dataset"],
                    "type": "dialogue",
                    "split": item["split"]
                }
            })

        # SOAP chunks
        n_chunks = splitter.split_text(item["output"])
        for ch in n_chunks:
            docs.append({
                "text": ch,
                "metadata": {
                    "id": item["id"],
                    "dataset": item["dataset"],
                    "type": "note",
                    "split": item["split"]
                }
            })

    return docs


# ----------------------------------------------------------
# Build FAISS Index
# ----------------------------------------------------------
def build_faiss_index(docs, out_path, embedding_model):

    print(f"🔹 Loading embeddings: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    texts = [d["text"] for d in docs]
    metadata = [d["metadata"] for d in docs]

    print("🔹 Embedding & Indexing...")
    db = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadata
    )

    os.makedirs(out_path, exist_ok=True)
    db.save_local(out_path)
    print(f"✔ FAISS index saved → {out_path}")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out_index", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()

    print("📥 Loading dataset...")
    data = load_examples(args.infile)

    print("🔪 Chunking text...")
    docs = build_chunks(
        data,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    print(f"📌 Total chunks created: {len(docs)}")

    print("📦 Building FAISS index...")
    build_faiss_index(docs, args.out_index, args.embedding_model)


if __name__ == "__main__":
    main()
