"""
DSCI-12: Clean Amazon reviews and run the baseline ABSA pipeline.

Usage:
    python prepare_data.py data/amazon_reviews_cleaned.csv

Outputs beside the input file:
    absa_ready.csv
    aspect_mentions.csv
    to_label_sample.csv

Baseline:
    - keyword-based aspect extraction
    - TF-IDF + Logistic Regression sentiment model
    - star ratings used as weak labels
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


SRC = sys.argv[1] if len(sys.argv) > 1 else "amazon_reviews_cleaned.csv"
OUT = os.path.dirname(os.path.abspath(SRC))


# ---------- 1. Load and clean reviews ----------
df = pd.read_csv(SRC)

required = {"reviewText", "reviewTime", "overall"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

n0 = len(df)

df["review_text"] = (
    df["reviewText"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.replace(r"([a-zA-Z0-9])([.!?])([A-Z])", r"\1\2 \3", regex=True)
)

df = df[df["review_text"].ne("")]
df = df[~df["review_text"].str.lower().duplicated()]
df = df[df["review_text"].str.split().str.len() >= 3].copy()
df = df.reset_index(drop=True)

df["date"] = pd.to_datetime(df["reviewTime"], errors="coerce")
df["quarter"] = df["date"].dt.to_period("Q").astype("string").fillna("Unknown")
df["rating"] = pd.to_numeric(df["overall"], errors="coerce")
df = df[df["rating"].between(1, 5)].copy()
df["rating"] = df["rating"].astype(int)
df = df.reset_index(drop=True)

df["star_label"] = np.where(
    df["rating"] >= 4,
    "positive",
    np.where(df["rating"] == 3, "neutral", "negative"),
)
df["review_id"] = np.arange(len(df), dtype=int)

# These are optional in some Amazon review exports.
for col in ("helpful_yes", "total_vote"):
    if col not in df.columns:
        df[col] = 0


# A rough segment inferred from the first device/category keyword in the review.
SEG = {
    "Phone": r"phone|galaxy|samsung|droid|htc|android|\bs[345]\b|\bnote ?[23]\b|\blumia\b|iphone",
    "Tablet": r"tablet|nexus|ipad|kindle|nook|\btab\b",
    "Action cam (GoPro)": r"gopro|go pro|action cam",
    "Camera": r"camera|dslr|canon|nikon|sony|camcorder",
    "Laptop": r"laptop|notebook|computer|\bpc\b|macbook",
}


def infer_segment(text):
    text = str(text).lower()
    first_match = (float("inf"), None)
    for name, pattern in SEG.items():
        match = re.search(pattern, text, flags=re.I)
        if match and match.start() < first_match[0]:
            first_match = (match.start(), name)
    return first_match[1]


df["segment"] = df["review_text"].map(infer_segment)

ready_cols = [
    "review_id", "rating", "star_label", "review_text",
    "date", "quarter", "segment", "helpful_yes", "total_vote",
]
df[ready_cols].to_csv(
    os.path.join(OUT, "absa_ready.csv"),
    index=False,
    encoding="utf-8",
)
print(f"Cleaned: {n0} -> {len(df)} reviews")


# ---------- 2. Aspect extraction (keyword baseline) ----------
ASP = {
    "speed": r"fast|speed|slow|transfer|mb/s|class ?10|lag|quick|read|write|loading",
    "storage": r"storage|space|capacity|\bgb\b|room|memory|fills?|full",
    "price": r"price|cheap|value|money|cost|deal|bargain|worth|\$|afford|pricey|expensive",
    "reliability": r"\bwork(?:s|ed|ing)?\b|fail|died|dead|stopped|corrupt|defective|reliab|error|crash|broke|lost|issue|problem",
    "compatibility": r"compatib|recogni[sz]|detect|format|adapter|fits?\b|works? with|slot|support",
    "video & photo": r"video|record|1080|4k|photo|picture|\bhd\b|footage|shoot",
    "delivery": r"ship|deliver|arriv|packag|seller|received|box",
    "authenticity": r"fake|counterfeit|genuine|authentic|legit|real sandisk|knock",
}


def split_clauses(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    result = []
    for sentence in sentences:
        pieces = re.split(
            r",?\s+(?:but|however|although|though|except|yet)\s+",
            sentence,
            flags=re.I,
        )
        result.extend(piece.strip() for piece in pieces if len(piece.split()) >= 2)
    return result


rows = []
for review in df.itertuples(index=False):
    for clause in split_clauses(review.review_text):
        hits = []
        for aspect, pattern in ASP.items():
            match = re.search(pattern, clause, flags=re.I)
            if match:
                hits.append((aspect, match.group(0)))
        rows.append({
            "review_id": review.review_id,
            "clause": clause,
            "aspects": hits,
        })

cl = pd.DataFrame(rows, columns=["review_id", "clause", "aspects"])
if cl.empty:
    raise ValueError("No review clauses were produced. Check the input reviewText column.")

cl = cl.merge(
    df[["review_id", "rating", "segment", "quarter"]],
    on="review_id",
    how="left",
)

# 1 = positive, 0 = negative, -1 = neutral/ignored during training.
cl["y"] = np.where(cl["rating"] >= 4, 1, np.where(cl["rating"] <= 2, 0, -1))
cl["p"] = np.nan

# Split by review ID to avoid putting clauses from one review in both train and test.
unique_ids = df["review_id"].to_numpy()
n_splits = min(5, len(unique_ids))
if n_splits < 2:
    raise ValueError("At least two reviews are needed for cross-validation.")

folds = KFold(n_splits=n_splits, shuffle=True, random_state=7)

for train_idx, test_idx in folds.split(unique_ids):
    train_ids = set(unique_ids[train_idx])
    test_ids = set(unique_ids[test_idx])

    train_mask = cl["review_id"].isin(train_ids) & cl["y"].isin([0, 1])
    test_mask = cl["review_id"].isin(test_ids)

    train_text = cl.loc[train_mask, "clause"]
    train_y = cl.loc[train_mask, "y"]

    if train_y.nunique() < 2 or train_mask.sum() == 0:
        # Keep NaN probabilities if a fold lacks both classes.
        continue

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(cl.loc[test_mask, "clause"])

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X_train, train_y)

    positive_index = list(model.classes_).index(1)
    cl.loc[test_mask, "p"] = model.predict_proba(X_test)[:, positive_index]

cl["sent"] = np.where(
    cl["p"].isna(),
    "neutral",
    np.where(cl["p"] >= 0.65, "positive",
             np.where(cl["p"] <= 0.35, "negative", "neutral")),
)

mentions = cl[cl["aspects"].map(bool)].explode("aspects").copy()
mentions = mentions.dropna(subset=["aspects"]).reset_index(drop=True)

if mentions.empty:
    mentions_out = pd.DataFrame(columns=[
        "review_id", "aspect", "keyword", "clause", "sentiment",
        "prob_positive", "segment", "quarter", "rating",
    ])
else:
    mentions["aspect"] = mentions["aspects"].map(lambda x: x[0])
    mentions["keyword"] = mentions["aspects"].map(lambda x: str(x[1]).lower())
    mentions["quarter"] = mentions["quarter"].astype(str)
    mentions_out = mentions[
        ["review_id", "aspect", "keyword", "clause", "sent", "p",
         "segment", "quarter", "rating"]
    ].rename(columns={"sent": "sentiment", "p": "prob_positive"})
    mentions_out["prob_positive"] = mentions_out["prob_positive"].round(3)

mentions_path = os.path.join(OUT, "aspect_mentions.csv")
mentions_out.to_csv(mentions_path, index=False, encoding="utf-8")

if not mentions_out.empty:
    print(
        "Mentions:", len(mentions_out),
        "| sentiment distribution:",
        mentions_out["sentiment"].value_counts(normalize=True).round(2).to_dict(),
    )

    eval_rows = mentions_out[
        mentions_out["rating"].ne(3) & mentions_out["prob_positive"].notna()
    ]
    if not eval_rows.empty:
        true_labels = np.where(eval_rows["rating"] >= 4, "positive", "negative")
        predicted_labels = eval_rows["sentiment"].replace({"neutral": "positive"})
        print(
            "Proxy check vs stars (not human ground truth): "
            f"acc={accuracy_score(true_labels, predicted_labels):.3f} "
            f"macroF1={f1_score(true_labels, predicted_labels, average='macro'):.3f}"
        )
else:
    print("No aspect mentions matched the keyword rules.")

sample_path = os.path.join(OUT, "to_label_sample.csv")
if os.path.exists(sample_path):
    print("to_label_sample.csv already exists; left untouched.")
elif not mentions_out.empty:
    sample_size = min(300, len(mentions_out))
    mentions_out.sample(sample_size, random_state=3)[
        ["review_id", "aspect", "clause"]
    ].assign(true_sentiment="").to_csv(
        sample_path,
        index=False,
        encoding="utf-8",
    )
    print(f"Created {sample_path} with {sample_size} rows to label manually.")

print("Baseline outputs saved in:", OUT)
