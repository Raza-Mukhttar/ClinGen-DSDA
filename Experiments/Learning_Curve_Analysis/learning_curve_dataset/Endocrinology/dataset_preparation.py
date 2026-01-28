from datasets import load_dataset
import pandas as pd
import os

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "/datasets/learning_curve/Endocrinology"
PERCENTAGES = [20, 30, 50, 60, 70, 80, 90]
RANDOM_SEED = 42

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("starfishdata/endocrinology_structured_notes_1500")

# Convert to pandas
df = pd.DataFrame(dataset["train"])

# Shuffle once (critical for learning curves)
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

total_size = len(df)
print(f"Total samples: {total_size}")

# -----------------------------
# Create cumulative training splits
# -----------------------------
for p in PERCENTAGES:
    subset_size = int((p / 100) * total_size)
    subset_df = df.iloc[:subset_size]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"train_{p}pct.csv"
    )

    subset_df.to_csv(output_path, index=False)

    print(f"Saved {output_path} -> {subset_size} samples")
