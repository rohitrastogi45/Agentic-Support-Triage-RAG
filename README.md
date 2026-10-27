# Agentic Support Triage RAG — Memory-Efficient EDA

This starter project performs streaming EDA on the Customer Support on Twitter dataset. It defaults to 50,000-row chunks and only samples a bounded number of inbound messages for text/sentiment analysis.

## 1. Put the dataset in place

Download `twcs.csv` from Kaggle and place it here:

`data/raw/twcs.csv`

Do not commit the raw dataset or generated partitions to GitHub.

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the VS Code terminal with Command Prompt and run `.venv\Scripts\activate.bat`, or invoke `.venv\Scripts\python.exe` directly.

## 3. Split rows for two members

From the project root:

```powershell
python src/split_dataset.py --input data/raw/twcs.csv --chunk-size 50000
```

Outputs: `data/processed/member1_data.csv` and `member2_data.csv`.

The split uses a stable hash of `tweet_id`. Missing IDs are assigned by source row number. This ensures each input row is assigned to exactly one partition. Identical nonmissing IDs are routed to the same partition; duplicate records are not removed. The partition sizes should be approximately, but not necessarily exactly, equal.

If disk space is limited, skip creating two copies and instead assign team members complementary row ranges, but adapt the scripts to read the original file and filter those ranges.

## 4. Run Member 1 EDA

```powershell
python src/run_eda.py --member 1 --chunk-size 50000
```

## 5. Run Member 2 EDA

```powershell
python src/run_eda.py --member 2 --chunk-size 50000 --sample-size 15000 --seed 42
```

Member 2's customer text, word counts, frequent words/bigrams, and VADER sentiment are based on a uniform reservoir sample of nonempty inbound messages, not the full partition.

## 6. Combine reports

After both members finish:

```powershell
python src/combine_reports.py
```

## 7. RAM safety

- Start with `--chunk-size 25000` if Windows memory use becomes high.
- Do not open the large CSV partitions in Excel.
- Close browser tabs and unused applications before running the scripts.
- Keep the laptop on a hard, ventilated surface.
- SQLite is used temporarily for exact duplicate-ID scanning so millions of IDs do not need to remain in Python memory. This can take time and disk I/O.
- The script never loads a complete partition into a DataFrame.
- Text vectorization and VADER are limited to the configured sample.

## 8. Outputs

Member-specific reports are saved under `reports/` with prefixes `member1_` and `member2_`. The scripts produce CSV summaries and PNG charts. The exact filenames depend on which optional analyses are applicable.

## 9. Interpretation cautions

- `inbound=True` indicates an incoming tweet, not necessarily a complaint.
- Historical support tweets are not an approved knowledge base.
- Sentiment is not urgency or priority.
- Conversation references can point to tweets in the other partition.
- Do not average partition medians or percentiles to derive combined medians/percentiles.

## 10. Suggested team workflow

1. One person verifies dataset path and runs split script.
2. Both members run EDA on their partition.
3. Compare data quality and inbound/outbound distributions.
4. Member 2's text sample findings are treated as exploratory.
5. Run the report-combination script.
6. Write the final narrative from actual CSVs and plots; never invent findings.
7. Continue with issue-category labeling and a separate, curated set of trusted FAQ/policy documents for RAG.
