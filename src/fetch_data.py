"""Download the MMLU test split and save a fixed, stratified subset as JSON.

The subset is what every serving configuration is measured on (src/run_configs.py) and what
the study sessions draw their questions from (src/build_session.py).

Usage: python3 src/fetch_data.py [--per-subject 20] [--seed 0]
Source: Hendrycks et al. (2021), "Measuring Massive Multitask Language Understanding",
        hosted at huggingface.co/datasets/cais/mmlu (MIT licence).
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mmlu_subset.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)

    rng = random.Random(args.seed)
    items = []
    for subject, group in sorted(df.groupby("subject")):
        rows = group.to_dict("records")
        for row in rng.sample(rows, min(args.per_subject, len(rows))):
            items.append({
                "id": f"{subject}-{len(items):04d}",
                "subject": subject,
                "question": row["question"].strip(),
                "choices": [str(c).strip() for c in row["choices"]],
                "answer": "ABCD"[int(row["answer"])],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"wrote {len(items)} items from {df['subject'].nunique()} subjects to {OUT}")


if __name__ == "__main__":
    main()
