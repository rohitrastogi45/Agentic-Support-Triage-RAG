"""Split twcs.csv into two deterministic, memory-efficient partitions.

Usage (from project root):
    python src/split_dataset.py --input data/raw/twcs.csv --chunk-size 50000
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd


def bucket_for(value: object, row_number: int) -> int:
    """Stable 0/1 bucket. Missing IDs use row number so rows are not dropped."""
    if pd.isna(value) or str(value).strip() == "":
        key = f"missing-id-row:{row_number}"
    else:
        key = f"tweet-id:{str(value).strip()}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/twcs.csv")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source.resolve()}\nPut twcs.csv in data/raw/ or pass --input.")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out1, out2 = out_dir / "member1_data.csv", out_dir / "member2_data.csv"
    # Avoid accidentally appending new runs to old partitions.
    for path in (out1, out2):
        if path.exists():
            path.unlink()

    total = 0
    counts = [0, 0]
    missing_ids = 0
    first_write = [True, True]
    reader = pd.read_csv(source, chunksize=args.chunk_size, low_memory=False)
    for chunk in reader:
        if "tweet_id" not in chunk.columns:
            raise ValueError("Expected column 'tweet_id' was not found. Check the CSV header.")
        assignments = []
        for local_i, value in enumerate(chunk["tweet_id"].tolist()):
            global_i = total + local_i
            if pd.isna(value) or str(value).strip() == "":
                missing_ids += 1
            assignments.append(bucket_for(value, global_i))
        assignment = pd.Series(assignments, index=chunk.index)
        for bucket, path in enumerate((out1, out2)):
            part = chunk.loc[assignment == bucket]
            if not part.empty:
                part.to_csv(path, mode="w" if first_write[bucket] else "a",
                            header=first_write[bucket], index=False)
                first_write[bucket] = False
                counts[bucket] += len(part)
        total += len(chunk)
        print(f"Processed {total:,} rows | Member 1: {counts[0]:,} | Member 2: {counts[1]:,}")

    if total == 0:
        raise ValueError("The CSV contains no data rows.")
    if counts[0] + counts[1] != total:
        raise RuntimeError(f"Split verification failed: input={total}, output={counts[0] + counts[1]}")
    print("\nSPLIT COMPLETE")
    print(f"Input rows: {total:,}")
    print(f"Member 1 rows: {counts[0]:,} ({counts[0] / total:.2%})")
    print(f"Member 2 rows: {counts[1]:,} ({counts[1] / total:.2%})")
    print(f"Rows with missing tweet_id (assigned using row number): {missing_ids:,}")
    print(f"Output rows sum to input rows: {counts[0] + counts[1] == total}")
    print("Note: identical non-missing tweet IDs always go to the same partition. Duplicate IDs are not removed.")
    print("Files:", out1, "and", out2)

if __name__ == "__main__":
    main()
