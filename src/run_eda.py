"""Memory-conscious EDA for either partition.

Usage:
  python src/run_eda.py --member 1
  python src/run_eda.py --member 2 --chunk-size 50000 --sample-size 15000
"""
from __future__ import annotations
import argparse, gc, hashlib, re, sqlite3, random
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

CHUNK_SIZE = 50_000
MAX_TEXT_SAMPLE = 15_000
RANDOM_SEED = 42


def normalize_inbound(series: pd.Series) -> pd.Series:
    """Normalize common CSV representations into True/False/<missing>/<unexpected>."""
    s = series.astype("string").str.strip().str.lower()
    mapping = {"true": "inbound", "1": "inbound", "1.0": "inbound", "yes": "inbound",
               "false": "outbound", "0": "outbound", "0.0": "outbound", "no": "outbound"}
    return s.map(mapping).fillna(s.where(s.notna(), "<missing>").replace("", "<missing>").map(lambda x: "<missing>" if x == "<missing>" else f"unexpected:{x}"))


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dirs() -> None:
    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("data/processed").mkdir(exist_ok=True, parents=True)


def save_bar(labels, values, title, ylabel, outpath, rotation=25):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(x) for x in labels], values)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=rotation)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def duplicate_scan(path: Path, chunksize: int, id_column: str) -> dict:
    """Exact duplicate-ID scan with SQLite on disk instead of a large Python set."""
    db_path = Path("data/processed") / f".duplicate_scan_member_{path.stem}.sqlite"
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE ids (tweet_id TEXT PRIMARY KEY)")
    missing = duplicate_occurrences = rows_with_id = 0
    try:
        for chunk in pd.read_csv(path, usecols=[id_column], chunksize=chunksize, low_memory=False):
            ids = chunk[id_column]
            missing += int(ids.isna().sum())
            vals = (str(v).strip() for v in ids.dropna().tolist() if str(v).strip() != "")
            for value in vals:
                rows_with_id += 1
                cur = conn.execute("INSERT OR IGNORE INTO ids(tweet_id) VALUES (?)", (value,))
                if cur.rowcount == 0:
                    duplicate_occurrences += 1
            conn.commit()
    finally:
        conn.close()
        try: db_path.unlink()
        except OSError: pass
    return {"rows_with_nonmissing_id": rows_with_id, "missing_tweet_id_count": missing,
            "duplicate_tweet_id_occurrences_after_first": duplicate_occurrences}


def reservoir_customer_sample(path: Path, chunksize: int, sample_size: int, seed: int) -> pd.DataFrame:
    """Uniform reservoir sample among nonempty inbound texts with bounded sample memory."""
    rng = random.Random(seed)
    reservoir = []
    eligible_seen = 0
    cols = ["tweet_id", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id", "inbound"]
    # Some datasets may omit optional fields; validate and select only available columns.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in cols if c in header]
    if "inbound" not in usecols or "text" not in usecols:
        raise ValueError("Member 2 text analysis requires 'inbound' and 'text' columns.")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        inbound = normalize_inbound(chunk["inbound"])
        texts = chunk["text"]
        eligible = chunk.loc[(inbound == "inbound") & texts.notna() & texts.astype("string").str.strip().ne("")]
        for record in eligible.to_dict(orient="records"):
            eligible_seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                j = rng.randrange(eligible_seen)
                if j < sample_size:
                    reservoir[j] = record
        del chunk, eligible
    sample = pd.DataFrame(reservoir, columns=usecols)
    sample["clean_text"] = sample["text"].map(clean_text)
    sample = sample.loc[sample["clean_text"].str.len() > 0].reset_index(drop=True)
    sample.to_csv("data/processed/member2_customer_sample.csv", index=False)
    print(f"Eligible inbound non-empty messages seen: {eligible_seen:,}; sample saved: {len(sample):,}")
    return sample


def run(member: int, chunksize: int, sample_size: int, seed: int) -> None:
    ensure_dirs()
    path = Path(f"data/processed/member{member}_data.csv")
    if not path.exists():
        raise FileNotFoundError(f"Partition not found: {path}. Run src/split_dataset.py first.")
    prefix = f"member{member}_"
    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()
    expected = {"tweet_id", "inbound", "created_at", "text"}
    missing_required = sorted(expected - set(columns))
    if missing_required:
        raise ValueError(f"Missing expected columns: {missing_required}. Found: {columns}")

    # Full-partition streaming summary.
    total_rows = 0
    missing_counts = Counter()
    inbound_counts = Counter()
    monthly_counts = Counter()
    earliest = latest = None
    invalid_timestamps = 0
    inferred_dtypes = None
    response_id_nonmissing = in_reply_nonmissing = 0
    text_nonmissing = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if inferred_dtypes is None:
            inferred_dtypes = {c: str(chunk[c].dtype) for c in columns}
        total_rows += len(chunk)
        missing_counts.update({c: int(chunk[c].isna().sum()) for c in columns})
        inbound_counts.update(normalize_inbound(chunk["inbound"]).value_counts(dropna=False).to_dict())
        dates = pd.to_datetime(chunk["created_at"], errors="coerce", utc=True, format="mixed")
        invalid_timestamps += int(dates.isna().sum())
        valid = dates.dropna()
        if not valid.empty:
            lo, hi = valid.min(), valid.max()
            earliest = lo if earliest is None or lo < earliest else earliest
            latest = hi if latest is None or hi > latest else latest
            month_values = valid.dt.strftime("%Y-%m")
            monthly_counts.update(month_values.value_counts().to_dict())
        if "response_tweet_id" in chunk:
            response_id_nonmissing += int(chunk["response_tweet_id"].notna().sum())
        if "in_response_to_tweet_id" in chunk:
            in_reply_nonmissing += int(chunk["in_response_to_tweet_id"].notna().sum())
        text_nonmissing += int(chunk["text"].notna().sum())
        del chunk, dates
    gc.collect()

    # Dataset overview includes row count, column names, dtypes, file size and non-null counts.
    overview = pd.DataFrame({"column": columns,
                             "inferred_dtype_first_chunk": [inferred_dtypes.get(c, "unknown") if inferred_dtypes else "unknown" for c in columns],
                             "non_null_count": [total_rows - missing_counts[c] for c in columns],
                             "missing_count": [missing_counts[c] for c in columns],
                             "missing_percentage": [round(missing_counts[c] / total_rows * 100, 4) if total_rows else 0 for c in columns]})
    overview["total_rows"] = total_rows
    overview["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
    overview.to_csv(f"reports/{prefix}dataset_overview.csv", index=False)

    missing_df = pd.DataFrame({"column": columns, "missing_count": [missing_counts[c] for c in columns],
                               "missing_percentage": [round(missing_counts[c] / total_rows * 100, 4) if total_rows else 0 for c in columns],
                               "total_rows": total_rows})
    missing_df.to_csv(f"reports/{prefix}missing_values.csv", index=False)
    save_bar(missing_df["column"], missing_df["missing_percentage"], f"{prefix}Missing values by column", "Missing (%)", f"reports/{prefix}missing_values.png")

    inbound_df = pd.DataFrame(sorted(inbound_counts.items()), columns=["message_type", "count"])
    inbound_df["percentage"] = inbound_df["count"] / total_rows * 100 if total_rows else 0
    inbound_df.to_csv(f"reports/{prefix}inbound_outbound.csv", index=False)
    save_bar(inbound_df["message_type"], inbound_df["count"], f"{prefix}Incoming/outgoing messages", "Rows", f"reports/{prefix}inbound_outbound.png")

    time_df = pd.DataFrame(sorted(monthly_counts.items()), columns=["month_utc", "message_count"])
    time_df.to_csv(f"reports/{prefix}time_activity.csv", index=False)
    if not time_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(time_df["month_utc"], time_df["message_count"])
        ax.set_title(f"{prefix}Messages by month (UTC)"); ax.set_xlabel("Month"); ax.set_ylabel("Messages")
        ax.tick_params(axis="x", labelrotation=60); fig.tight_layout()
        fig.savefig(f"reports/{prefix}time_activity.png", dpi=150); plt.close(fig)

    # Exact duplicate IDs via temporary disk-backed SQLite.
    dup = duplicate_scan(path, chunksize, "tweet_id")
    dup.update({"partition": f"member{member}", "total_rows": total_rows,
                "earliest_timestamp_utc": str(earliest) if earliest is not None else "",
                "latest_timestamp_utc": str(latest) if latest is not None else "",
                "invalid_or_missing_timestamps": invalid_timestamps,
                "rows_with_response_tweet_id": response_id_nonmissing,
                "rows_with_in_response_to_tweet_id": in_reply_nonmissing,
                "rows_with_nonmissing_text": text_nonmissing})
    pd.DataFrame([dup]).to_csv(f"reports/{prefix}duplicate_analysis.csv", index=False)

    print(f"\n{prefix.upper()} EDA COMPLETE")
    print(f"Rows: {total_rows:,}; columns: {len(columns)}; file size: {path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Earliest valid timestamp (UTC): {earliest}; latest: {latest}")
    print(f"Invalid/missing timestamps: {invalid_timestamps:,}")
    print(f"Reports saved under reports/{prefix}*")

    if member == 2:
        sample = reservoir_customer_sample(path, chunksize, sample_size, seed)
        if sample.empty:
            print("No nonempty inbound text sample available; skipping text analytics.")
            return
        # Message lengths on bounded sample only.
        sample["character_count"] = sample["clean_text"].str.len()
        sample["word_count"] = sample["clean_text"].str.split().str.len()
        stats = pd.DataFrame([{
            "sample_size": len(sample),
            "mean_word_count": sample["word_count"].mean(),
            "median_word_count": sample["word_count"].median(),
            "min_word_count": sample["word_count"].min(),
            "max_word_count": sample["word_count"].max(),
            "mean_character_count": sample["character_count"].mean(),
            "p25_word_count": sample["word_count"].quantile(.25),
            "p75_word_count": sample["word_count"].quantile(.75),
            "p90_word_count": sample["word_count"].quantile(.90),
        }])
        stats.to_csv("reports/member2_message_length.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5)); ax.hist(sample["word_count"], bins=40)
        ax.set_title("Customer message word-count distribution (sample)"); ax.set_xlabel("Words"); ax.set_ylabel("Messages")
        fig.tight_layout(); fig.savefig("reports/member2_message_length.png", dpi=150); plt.close(fig)

        docs = sample["clean_text"].astype(str).tolist()
        for kind, ngram, top_n, out_csv, out_png, title in [
            ("words", (1, 1), 30, "reports/member2_top_words.csv", "reports/member2_top_words.png", "Top words in customer-message sample"),
            ("bigrams", (2, 2), 20, "reports/member2_top_bigrams.csv", "reports/member2_top_bigrams.png", "Top bigrams in customer-message sample")]:
            vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram, max_features=1000, max_df=1.0, min_df=2)
            try:
                matrix = vectorizer.fit_transform(docs)
                totals = np.asarray(matrix.sum(axis=0)).ravel()
                terms = vectorizer.get_feature_names_out()
                order = np.argsort(totals)[::-1][:top_n]
                result = pd.DataFrame({"term": terms[order], "count": totals[order].astype(int)})
            except ValueError as exc:
                print(f"Skipping {kind}: {exc}")
                result = pd.DataFrame(columns=["term", "count"])
            result.to_csv(out_csv, index=False)
            if not result.empty:
                save_bar(result["term"], result["count"], title, "Count", out_png, rotation=60)
            del vectorizer
            gc.collect()

        # VADER is optional at import time so the rest of EDA can still run if missing.
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            sentiment_counts = Counter()
            for text in sample["clean_text"].astype(str):
                score = analyzer.polarity_scores(text)["compound"]
                label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
                sentiment_counts[label] += 1
            sent = pd.DataFrame([{"sentiment": label, "count": sentiment_counts.get(label, 0),
                                  "percentage": sentiment_counts.get(label, 0) / len(sample) * 100,
                                  "sample_size": len(sample)} for label in ["positive", "neutral", "negative"]])
            sent.to_csv("reports/member2_sentiment_summary.csv", index=False)
            save_bar(sent["sentiment"], sent["count"], "VADER sentiment in customer-message sample", "Messages", "reports/member2_sentiment_distribution.png")
        except ImportError:
            print("vaderSentiment not installed; run: pip install vaderSentiment")

        # Link field coverage only; no full-dataset join.
        links = pd.DataFrame([
            {"metric": "rows_with_response_tweet_id", "count": response_id_nonmissing},
            {"metric": "rows_with_in_response_to_tweet_id", "count": in_reply_nonmissing},
        ])
        links.to_csv("reports/member2_conversation_link_coverage.csv", index=False)
        print("Member 2 sample text analyses saved. Sentiment and text statistics describe the sample, not all tweets.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=int, choices=[1, 2], required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--sample-size", type=int, default=MAX_TEXT_SAMPLE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.sample_size < 1:
        raise ValueError("chunk-size and sample-size must be positive integers")
    run(args.member, args.chunk_size, args.sample_size, args.seed)

if __name__ == "__main__":
    main()
