"""DSCI-12 step 3: score a model against YOUR hand labels.
Usage: python evaluate.py [data_folder]        (default: data)
Needs: to_label_sample.csv with the true_sentiment column filled in
       (positive / neutral / negative, or wrong_aspect if the clause is not about that aspect)
       aspect_mentions.csv produced by the baseline or by your own model."""
import sys, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
D = sys.argv[1] if len(sys.argv) > 1 else "data"
lab = pd.read_csv(f"{D}/to_label_sample.csv")
lab["true_sentiment"] = lab["true_sentiment"].astype(str).str.lower().str.strip().replace({"nan": ""})
lab = lab[lab.true_sentiment != ""]
if lab.empty: sys.exit("No labels found. Fill in the true_sentiment column of to_label_sample.csv first.")
bad = lab[~lab.true_sentiment.isin(["positive", "neutral", "negative", "wrong_aspect"])]
if len(bad): sys.exit("Unknown labels (use positive/neutral/negative/wrong_aspect):\n" + str(bad.true_sentiment.value_counts()))
pred = pd.read_csv(f"{D}/aspect_mentions.csv")[["review_id", "aspect", "clause", "sentiment"]].drop_duplicates(["review_id", "aspect", "clause"])
x = lab.merge(pred, on=["review_id", "aspect", "clause"], how="left")
print(f"labelled rows: {len(lab)} | found in model output: {x.sentiment.notna().sum()}")
wa = (x.true_sentiment == "wrong_aspect").mean()
print(f"aspect tagging: {100*(1-wa):.1f}% of labelled mentions were really about that aspect (wrong_aspect rate {100*wa:.1f}%)")
s = x[(x.true_sentiment != "wrong_aspect") & x.sentiment.notna()]
print(f"\nSENTIMENT on {len(s)} mentions   accuracy={accuracy_score(s.true_sentiment, s.sentiment):.3f}   macro-F1={f1_score(s.true_sentiment, s.sentiment, average='macro'):.3f}\n")
print(classification_report(s.true_sentiment, s.sentiment, zero_division=0))
L = ["positive", "neutral", "negative"]
print("confusion matrix (rows = your label, columns = model):")
print(pd.DataFrame(confusion_matrix(s.true_sentiment, s.sentiment, labels=L), index=L, columns=L))
print("\nper aspect (accuracy):"); print(s.assign(ok=s.true_sentiment == s.sentiment).groupby("aspect").ok.agg(["mean", "count"]).round(2))
