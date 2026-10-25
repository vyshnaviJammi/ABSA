"""
DSCI-12: Transformer-based aspect sentiment inference.

This script keeps keyword-based aspect extraction but replaces the TF-IDF
sentiment classifier with an aspect-conditioned DeBERTa ABSA model.

Usage:
    pip install -U pandas numpy torch transformers sentencepiece
    python transformer_absa.py data/amazon_reviews_cleaned.csv

Outputs beside the input file:
    absa_ready.csv
    aspect_mentions.csv

Model:
    yangheng/deberta-v3-base-absa-v1.1

Notes:
    - The model downloads its weights on the first run (large download).
    - CPU inference works but may be slow; a compatible GPU helps.
    - Aspect extraction remains keyword-based in this script.
    - This is pretrained inference, not fine-tuning on your Amazon dataset.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import pipeline


SRC = sys.argv[1] if len(sys.argv) > 1 else "data/amazon_reviews_cleaned.csv"
OUT = os.path.dirname(os.path.abspath(SRC))
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"


# ---------- 1. Clean reviews (same output schema as baseline) ----------
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
    df["rating"] >= 4, "positive",
    np.where(df["rating"] == 3, "neutral", "negative"),
)
df["review_id"] = np.arange(len(df), dtype=int)

for col in ("helpful_yes", "total_vote"):
    if col not in df.columns:
        df[col] = 0

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
df[[
    "review_id", "rating", "star_label", "review_text", "date",
    "quarter", "segment", "helpful_yes", "total_vote",
]].to_csv(os.path.join(OUT, "absa_ready.csv"), index=False, encoding="utf-8")
print(f"Cleaned: {n0} -> {len(df)} reviews")


# ---------- 2. Find aspect mentions using the same keyword rules ----------
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
    clauses = []
    for sentence in sentences:
        parts = re.split(
            r",?\s+(?:but|however|although|though|except|yet)\s+",
            sentence,
            flags=re.I,
        )
        clauses.extend(part.strip() for part in parts if len(part.split()) >= 2)
    return clauses


rows = []
for review in df.itertuples(index=False):
    for clause in split_clauses(review.review_text):
        for aspect, pattern in ASP.items():
            match = re.search(pattern, clause, flags=re.I)
            if match:
                rows.append({
                    "review_id": review.review_id,
                    "aspect": aspect,
                    "keyword": match.group(0).lower(),
                    "clause": clause,
                    "segment": review.segment,
                    "quarter": str(review.quarter),
                    "rating": review.rating,
                })

mentions = pd.DataFrame(rows, columns=[
    "review_id", "aspect", "keyword", "clause", "segment", "quarter", "rating"
])
if mentions.empty:
    raise ValueError(
        "No aspect mentions matched. Check the data or expand the aspect keyword rules."
    )

# ---------- 3. Aspect-conditioned Transformer sentiment ----------
# The ABSA model receives both the clause and the aspect, so it can distinguish
# e.g. positive camera sentiment from negative battery sentiment in one review.
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=device,
)

pairs = [
    {"text": row.clause, "text_pair": row.aspect}
    for row in mentions.itertuples(index=False)
]

# top_k=None returns class scores for each clause/aspect pair.
# Chunking limits peak memory use.
sentiments = []
positive_scores = []
batch_size = 16 if device >= 0 else 8

for start in range(0, len(pairs), batch_size):
    batch = pairs[start:start + batch_size]
    outputs = classifier(batch, top_k=None, truncation=True, batch_size=batch_size)

    for output in outputs:
        # Depending on Transformers version, each item may be a list of labels/scores.
        if isinstance(output, dict):
            output = [output]
        scores = {item["label"].strip().lower(): float(item["score"]) for item in output}

        # Model card labels are Negative, Neutral, Positive.
        # Fail loudly rather than silently guessing if a different label schema appears.
        required_labels = {"negative", "neutral", "positive"}
        if not required_labels.issubset(scores):
            raise ValueError(
                "Unexpected ABSA model labels: "
                f"{sorted(scores)}. Inspect model.config.id2label before mapping."
            )

        best_label = max(required_labels, key=lambda label: scores[label])
        sentiments.append(best_label)
        positive_scores.append(scores["positive"])

mentions["sentiment"] = sentiments
mentions["prob_positive"] = np.round(positive_scores, 4)

output_cols = [
    "review_id", "aspect", "keyword", "clause", "sentiment",
    "prob_positive", "segment", "quarter", "rating",
]
mentions[output_cols].to_csv(
    os.path.join(OUT, "aspect_mentions.csv"),
    index=False,
    encoding="utf-8",
)

print("Transformer ABSA model:", MODEL_NAME)
print("Aspect mentions:", len(mentions))
print("Sentiment distribution:",
      mentions["sentiment"].value_counts(normalize=True).round(3).to_dict())
print("Outputs saved in:", OUT)
print("Reminder: evaluate against manually labelled aspect sentiment for a fair result.")
