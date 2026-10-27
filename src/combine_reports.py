"""Combine additive summary reports from both members without loading raw data."""
from pathlib import Path
import pandas as pd

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True, parents=True)


def read_csv(path):
    return pd.read_csv(path) if path.exists() else None

rows = []
for member in (1, 2):
    overview = read_csv(REPORTS / f"member{member}_dataset_overview.csv")
    inbound = read_csv(REPORTS / f"member{member}_inbound_outbound.csv")
    duplicate = read_csv(REPORTS / f"member{member}_duplicate_analysis.csv")
    if overview is not None and not overview.empty:
        rows.append({"member": member, "total_rows": int(overview["total_rows"].iloc[0]),
                     "file_size_mb": float(overview["file_size_mb"].iloc[0])})
    if inbound is not None:
        inbound.to_csv(REPORTS / f"member{member}_inbound_outbound_copy.csv", index=False)

if rows:
    pd.DataFrame(rows).to_csv(REPORTS / "final_eda_summary.csv", index=False)
else:
    print("No member overview reports found. Run EDA for members 1 and 2 first.")
    raise SystemExit(1)

# Correctly combine missing counts and denominators; percentages are recomputed.
missing_frames = []
for member in (1, 2):
    frame = read_csv(REPORTS / f"member{member}_missing_values.csv")
    if frame is not None:
        frame["member"] = member
        missing_frames.append(frame)
if missing_frames:
    all_missing = pd.concat(missing_frames, ignore_index=True)
    combined = all_missing.groupby("column", as_index=False).agg(missing_count=("missing_count", "sum"), total_rows=("total_rows", "sum"))
    combined["missing_percentage"] = (combined["missing_count"] / combined["total_rows"] * 100).round(4)
    combined.to_csv(REPORTS / "combined_missing_values.csv", index=False)

# Combine additive inbound/outbound counts and recalculate percentages.
frames = []
for member in (1, 2):
    frame = read_csv(REPORTS / f"member{member}_inbound_outbound.csv")
    if frame is not None:
        frames.append(frame[["message_type", "count"]])
if frames:
    combined_io = pd.concat(frames, ignore_index=True).groupby("message_type", as_index=False)["count"].sum()
    total = combined_io["count"].sum()
    combined_io["percentage"] = (combined_io["count"] / total * 100).round(4) if total else 0
    combined_io.to_csv(REPORTS / "combined_inbound_outbound.csv", index=False)

summary = """# Agentic Support Triage RAG — EDA Summary\n\nThis report is generated from the two member-level EDA outputs. Replace this paragraph with interpretations based on the saved CSV reports and charts. Do not claim sample-based text or sentiment results represent every tweet.\n\n## Important interpretation notes\n- The two partition counts should sum to the source row count.\n- Missing percentages are recomputed from combined counts and denominators.\n- Text, word, bigram, and sentiment results are based on Member 2's bounded customer-message sample.\n- Historical tweets are conversation examples, not verified support policies.\n- Sentiment is not the same as urgency or priority.\n\n## Output files\n- `final_eda_summary.csv`: partition row counts and file sizes.\n- `combined_missing_values.csv`: combined missing-value counts and percentages.\n- `combined_inbound_outbound.csv`: combined message-type counts and percentages.\n"""
(REPORTS / "eda_summary.md").write_text(summary, encoding="utf-8")
print("Combined reports written to reports/final_eda_summary.csv, combined_missing_values.csv, combined_inbound_outbound.csv, and eda_summary.md")

# Combine additive monthly activity counts.
time_frames = []
for member in (1, 2):
    frame = read_csv(REPORTS / f"member{member}_time_activity.csv")
    if frame is not None and not frame.empty:
        time_frames.append(frame[["month_utc", "message_count"]])
if time_frames:
    combined_time = pd.concat(time_frames, ignore_index=True).groupby("month_utc", as_index=False)["message_count"].sum()
    combined_time.to_csv(REPORTS / "combined_time_activity.csv", index=False)

# Keep partition duplicate scans separate; sum of duplicate occurrences is not necessarily
# a global duplicate count if an ID was duplicated in a way that could cross partitions.
dup_frames = []
for member in (1, 2):
    frame = read_csv(REPORTS / f"member{member}_duplicate_analysis.csv")
    if frame is not None and not frame.empty:
        dup_frames.append(frame)
if dup_frames:
    pd.concat(dup_frames, ignore_index=True).to_csv(REPORTS / "partition_duplicate_analysis.csv", index=False)
