
"""
DSCI-12 | Step 2: Build website data for the ABSA dashboard.

Reads:
    data/absa_ready.csv
    data/aspect_mentions.csv

Creates:
    data/site_data.json
    Updates index.html if it contains DATA_START / DATA_END markers.

Usage:
    python build_site.py
    python build_site.py data index.html
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data"
HTML_FILE = sys.argv[2] if len(sys.argv) > 2 else "index.html"

REVIEWS_FILE = os.path.join(DATA_FOLDER, "absa_ready.csv")
MENTIONS_FILE = os.path.join(DATA_FOLDER, "aspect_mentions.csv")
OUTPUT_FILE = os.path.join(DATA_FOLDER, "site_data.json")


# --------------------------------------------------
# 2. Load reviews and aspect-level sentiment data
# --------------------------------------------------

df = pd.read_csv(REVIEWS_FILE, parse_dates=["date"])
mentions = pd.read_csv(MENTIONS_FILE)

# Remove duplicate metadata columns before joining review details.
metadata_columns = ["segment", "quarter", "rating"]
mentions = mentions.drop(
    columns=[col for col in metadata_columns if col in mentions.columns]
)

mentions = mentions.merge(
    df[["review_id", "segment", "quarter", "rating"]],
    on="review_id",
    how="left",
)

# Use a shorter column name for sentiment.
mentions = mentions.rename(columns={"sentiment": "sent"})

# Keep only valid sentiment labels.
mentions["sent"] = mentions["sent"].astype(str).str.lower().str.strip()
mentions = mentions[
    mentions["sent"].isin(["positive", "neutral", "negative"])
].copy()

# Use the aspect name as the keyword if no keyword is provided.
if "keyword" not in mentions.columns:
    mentions["keyword"] = mentions["aspect"]
else:
    mentions["keyword"] = mentions["keyword"].fillna(mentions["aspect"])

mentions["keyword"] = mentions["keyword"].astype(str).str.lower()


# --------------------------------------------------
# 3. Prepare general review statistics
# --------------------------------------------------

# Keep the five most frequently occurring product segments.
top_segments = df["segment"].value_counts().head(5)
segments = list(top_segments.index)

# Use a fixed seed so the example reviews stay reproducible.
rng = np.random.RandomState(5)


def sentiment_summary(group):
    """Return the review count and sentiment percentages."""
    count = len(group)

    if count == 0:
        return {"n": 0, "positive": 0, "neutral": 0, "negative": 0}

    return {
        "n": count,
        "positive": round((group["sent"] == "positive").mean() * 100),
        "neutral": round((group["sent"] == "neutral").mean() * 100),
        "negative": round((group["sent"] == "negative").mean() * 100),
    }


# Store the overall statistics used by the dashboard.
data = {
    "n_reviews": len(df),
    "n_with_device": int(df["segment"].notna().sum()),
    "date_min": str(df["date"].min().date()),
    "date_max": str(df["date"].max().date()),
    "segments": [
        {"name": name, "n": int(count)}
        for name, count in top_segments.items()
    ],
    "aspects": {},
}

quarters = sorted(df["quarter"].dropna().unique().tolist())


# --------------------------------------------------
# 4. Build sentiment statistics for each aspect
# --------------------------------------------------

for aspect, group in mentions.groupby("aspect"):
    total = len(group)

    positive_pct = round((group["sent"] == "positive").mean() * 100)
    negative_pct = round((group["sent"] == "negative").mean() * 100)

    aspect_data = {
        "n": total,
        "pos": positive_pct,
        "neg": negative_pct,
        "neu": 100 - positive_pct - negative_pct,
    }

    # Sentiment distribution for each major segment.
    aspect_data["seg"] = {
        segment: sentiment_summary(group[group["segment"] == segment])
        for segment in segments
    }

    # Track positive sentiment over time.
    aspect_data["trend"] = []

    for quarter in quarters:
        quarter_group = group[group["quarter"] == quarter]

        if len(quarter_group) >= 8:
            positive = (quarter_group["sent"] == "positive").mean() * 100
            aspect_data["trend"].append(round(positive))
        else:
            aspect_data["trend"].append(None)

    # Select a few review excerpts for the dashboard.
    samples = []

    for sentiment, sample_count in [
        ("positive", 2),
        ("negative", 2),
        ("neutral", 1),
    ]:
        candidates = group[
            (group["sent"] == sentiment)
            & group["clause"].str.len().between(35, 220)
        ]

        # Randomly select up to the requested number of examples.
        selected_indices = rng.permutation(len(candidates))[:sample_count]

        for review in candidates.iloc[selected_indices].itertuples():
            samples.append({
                "s": sentiment,
                "seg": review.segment if isinstance(review.segment, str) else "",
                "t": review.clause,
                "kw": review.keyword,
            })

    aspect_data["samples"] = samples
    data["aspects"][str(aspect)] = aspect_data


# --------------------------------------------------
# 5. Define product names and matching keywords
# --------------------------------------------------
# A review is associated with a product when its text
# contains one of the product's matching patterns.

PRODUCT_PATTERNS = {
    "Samsung Galaxy S3": (
        ["galaxy s3", "s3", "siii"],
        r"\bs3\b|\bsiii\b|\bgs3\b|\bs iii\b|galaxy s3",
    ),
    "Samsung Galaxy S4": (
        ["galaxy s4", "s4"],
        r"\bs4\b|galaxy s4|\bs iv\b",
    ),
    "Samsung Galaxy Note": (
        ["galaxy note", "note"],
        r"galaxy note|note ?[23]\b|note ii|note 10\b|samsung note",
    ),
    "Samsung Galaxy Tab": (
        ["galaxy tab", "tab"],
        r"galaxy tab|\btab ?[234]\b|samsung tab",
    ),
    "Samsung Galaxy phone": (
        ["samsung", "galaxy"],
        r"samsung|galaxy",
    ),
    "Motorola phone": (
        ["motorola", "moto", "razr"],
        r"motorola|moto x|moto g|razr|droid x|droid bionic|droid 4|atrix|xoom",
    ),
    "HTC phone": (
        ["htc"],
        r"\bhtc\b",
    ),
    "LG phone": (
        ["lg"],
        r"\blg\b",
    ),
    "Nokia Lumia phone": (
        ["nokia", "lumia"],
        r"nokia|lumia",
    ),
    "BlackBerry phone": (
        ["blackberry"],
        r"blackberry",
    ),
    "Apple iPhone": (
        ["iphone"],
        r"iphone",
    ),
    "Apple MacBook": (
        ["macbook", "mac"],
        r"macbook",
    ),
    "GoPro camera": (
        ["gopro", "go pro", "hero"],
        r"gopro|go pro|hero ?[34]",
    ),
    "Canon camera": (
        ["canon"],
        r"canon",
    ),
    "Sony camera": (
        ["sony"],
        r"\bsony\b",
    ),
    "Nikon camera": (
        ["nikon"],
        r"nikon",
    ),
    "DSLR camera": (
        ["dslr"],
        r"dslr",
    ),
    "Dash cam": (
        ["dash cam", "dashcam"],
        r"dash ?cam",
    ),
    "Garmin GPS": (
        ["garmin", "gps"],
        r"garmin",
    ),
    "Nook tablet": (
        ["nook"],
        r"nook",
    ),
    "Asus tablet": (
        ["asus", "transformer"],
        r"asus|transformer|nexus 7",
    ),
    "Microsoft Surface": (
        ["surface"],
        r"surface (?:pro|rt|2|3|tablet)|microsoft surface",
    ),
    "Dell device": (
        ["dell", "venue"],
        r"\bdell\b|venue",
    ),
}


# --------------------------------------------------
# 6. Calculate product-level sentiment statistics
# --------------------------------------------------

# Convert review text to lowercase for keyword matching.
review_text = df["review_text"].fillna("").astype(str).str.lower()

data["products"] = {}

for product_name, (aliases, pattern) in PRODUCT_PATTERNS.items():

    # Find reviews that mention this product.
    matched_ids = set(
        df.loc[
            review_text.str.contains(pattern, regex=True, na=False),
            "review_id",
        ]
    )

    # Ignore products with too few matching reviews.
    if len(matched_ids) < 15:
        continue

    product_reviews = df[df["review_id"].isin(matched_ids)]
    product_mentions = mentions[mentions["review_id"].isin(matched_ids)]

    group = product_mentions

    positive_pct = round((group["sent"] == "positive").mean() * 100)
    negative_pct = round((group["sent"] == "negative").mean() * 100)

    product_data = {
        "alias": aliases,
        "n_reviews": len(matched_ids),
        "n_mentions": len(group),
        "avg_rating": round(float(product_reviews["rating"].mean()), 2),
        "pos": positive_pct,
        "neg": negative_pct,
        "neu": 100 - positive_pct - negative_pct,
        "aspects": {},
        "samples": [],
    }

    # Product sentiment trend by quarter.
    product_data["trend"] = []

    for quarter in quarters:
        quarter_group = group[group["quarter"] == quarter]

        if len(quarter_group) >= 8:
            positive = (quarter_group["sent"] == "positive").mean() * 100
            product_data["trend"].append(round(positive))
        else:
            product_data["trend"].append(None)

    # Keep aspect-level statistics when enough mentions exist.
    for aspect, aspect_group in group.groupby("aspect"):

        if len(aspect_group) < 5:
            continue

        product_data["aspects"][str(aspect)] = {
            "n": len(aspect_group),
            "pos": round((aspect_group["sent"] == "positive").mean() * 100),
            "neu": round((aspect_group["sent"] == "neutral").mean() * 100),
            "neg": round((aspect_group["sent"] == "negative").mean() * 100),
        }

    # Collect example review excerpts for each sentiment.
    for sentiment, sample_count in [
        ("positive", 2),
        ("negative", 2),
        ("neutral", 1),
    ]:
        candidates = group[
            (group["sent"] == sentiment)
            & group["clause"].str.len().between(35, 220)
        ]

        selected_indices = rng.permutation(len(candidates))[:sample_count]

        for review in candidates.iloc[selected_indices].itertuples():
            product_data["samples"].append({
                "s": sentiment,
                "a": review.aspect,
                "t": review.clause,
                "kw": review.keyword,
            })

    data["products"][product_name] = product_data


# Sort products by number of matching reviews.
data["products"] = dict(
    sorted(
        data["products"].items(),
        key=lambda item: -item[1]["n_reviews"],
    )
)

data["quarters"] = quarters


# --------------------------------------------------
# 7. Save dashboard data as JSON
# --------------------------------------------------

# Allow Unicode text and reject invalid JSON numbers.
json_text = json.dumps(
    data,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write(json_text)

print(f"Saved dashboard data: {OUTPUT_FILE}")


# --------------------------------------------------
# 8. Update the HTML file if it exists
# --------------------------------------------------

if os.path.exists(HTML_FILE):

    with open(HTML_FILE, "r", encoding="utf-8") as file:
        html = file.read()

    # Replace the existing embedded JSON between these markers.
    marker_pattern = r"/\*DATA_START\*/.*?/\*DATA_END\*/"
    replacement = (
        "/*DATA_START*/"
        + json_text.replace("</", "<\\/")
        + "/*DATA_END*/"
    )

    updated_html, replacements = re.subn(
        marker_pattern,
        lambda match: replacement,
        html,
        count=1,
        flags=re.DOTALL,
    )

    if replacements:
        with open(HTML_FILE, "w", encoding="utf-8") as file:
            file.write(updated_html)

        print(f"Updated HTML file: {HTML_FILE}")
    else:
        print(
            f"Warning: No DATA_START / DATA_END markers found in {HTML_FILE}. "
            "The HTML file was not changed."
        )

else:
    print(f"HTML file not found: {HTML_FILE}. JSON data was still saved.")


# --------------------------------------------------
# 9. Print a short summary
# --------------------------------------------------

print("Segments:", top_segments.to_dict())

print(
    "Aspects:",
    {aspect: details["n"] for aspect, details in data["aspects"].items()},
)

print("Products included:", len(data["products"]))
