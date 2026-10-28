# Agentic Support Triage RAG — EDA Summary

This report is generated from the two member-level EDA outputs. Replace this paragraph with interpretations based on the saved CSV reports and charts. Do not claim sample-based text or sentiment results represent every tweet.

## Important interpretation notes
- The two partition counts should sum to the source row count.
- Missing percentages are recomputed from combined counts and denominators.
- Text, word, bigram, and sentiment results are based on Member 2's bounded customer-message sample.
- Historical tweets are conversation examples, not verified support policies.
- Sentiment is not the same as urgency or priority.

## Output files
- `final_eda_summary.csv`: partition row counts and file sizes.
- `combined_missing_values.csv`: combined missing-value counts and percentages.
- `combined_inbound_outbound.csv`: combined message-type counts and percentages.
