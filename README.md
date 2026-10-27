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
