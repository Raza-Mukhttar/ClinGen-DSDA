import os
import pandas as pd

# ================= CONFIG =================
SOURCE_TRAIN_CSV = "/datasets/ACI-TaskB-TrainingSet.csv"
OUTPUT_DIR       = "/datasets/learning_curve"

SIZES = [10, 20, 30, 50, 67]
RANDOM_SEED = 42
# ==========================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load full training data
    df = pd.read_csv(SOURCE_TRAIN_CSV)

    # Safety checks
    assert "dialogue" in df.columns, "Missing 'dialogue' column"
    assert "note" in df.columns, "Missing 'note' column"
    assert len(df) >= max(SIZES), "Training set smaller than required subset size"

    # Deterministic shuffle ONCE
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Loaded {len(df)} total training samples")
    print("Preparing nested learning-curve subsets:")

    for n in SIZES:
        out_path = os.path.join(
            OUTPUT_DIR,
            f"ACI-TaskB-Train-{n}.csv"
        )

        subset = df.iloc[:n]
        subset.to_csv(out_path, index=False)

        print(f"  ✓ {n:>3} samples → {out_path}")

    print("\nAll CSV subsets prepared successfully.")
    print("IMPORTANT: Do NOT regenerate these files again.")


if __name__ == "__main__":
    main()
