"""DSCI-12 step 1: clean the Amazon reviews and run the BASELINE ABSA model.
Usage: python prepare_data.py data/amazon_reviews_cleaned.csv
Outputs (next to the input file): absa_ready.csv, aspect_mentions.csv, to_label_sample.csv
Baseline only: keyword aspects + weak-label sentiment (stars -> clause classifier).
To use a better model, replace section 2-3 (or write your own script) so that it produces
aspect_mentions.csv with the same columns, then run build_site.py (see docs/ADVANCING_THE_MODEL.md)."""
import re, sys, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

SRC = sys.argv[1] if len(sys.argv) > 1 else "amazon_reviews_cleaned.csv"
import os
OUT = os.path.dirname(os.path.abspath(SRC))  # outputs go next to the input file
# ---------- 1. clean ----------
df = pd.read_csv(SRC)
n0 = len(df)
df["review_text"] = (df["reviewText"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                     .str.replace(r"([a-z\)])([.!?])([A-Z])", r"\1\2 \3", regex=True))
df = df[~df["review_text"].str.lower().duplicated()]
df = df[df["review_text"].str.split().str.len() >= 3].reset_index(drop=True)
df["date"] = pd.to_datetime(df["reviewTime"])
df["quarter"] = df["date"].dt.to_period("Q").astype(str)
df["rating"] = df["overall"].astype(int)
df["star_label"] = np.where(df.rating >= 4, "positive", np.where(df.rating == 3, "neutral", "negative"))
df["review_id"] = df.index
# usage segment = first device named in the review (stand-in for product; file has no product id)
SEG = {"Phone": r"phone|galaxy|samsung|droid|htc|android|\bs[345]\b|\bnote ?[23]\b|\blumia|iphone",
       "Tablet": r"tablet|nexus|ipad|kindle|nook|\btab\b",
       "Action cam (GoPro)": r"gopro|go pro|action cam",
       "Camera": r"camera|dslr|canon|nikon|sony|camcorder",
       "Laptop": r"laptop|notebook|computer|\bpc\b|macbook"}
def seg(t):
    t = t.lower(); best = (10**9, None)
    for k, p in SEG.items():
        m = re.search(p, t)
        if m and m.start() < best[0]: best = (m.start(), k)
    return best[1]
df["segment"] = df["review_text"].map(seg)
df[["review_id", "rating", "star_label", "review_text", "date", "quarter", "segment", "helpful_yes", "total_vote"]] \
    .to_csv(f"{OUT}/absa_ready.csv", index=False)
print(f"cleaned: {n0} -> {len(df)} reviews")

# ---------- 2. aspects (lexicon baseline) ----------
ASP = {"speed": r"fast|speed|slow|transfer|mb/s|class ?10|lag|quick|read|write|loading",
       "storage": r"storage|space|capacity|\bgb\b|room|memory|fills?|full",
       "price": r"price|cheap|value|money|cost|deal|bargain|worth|\$|afford|pricey|expensive",
       "reliability": r"work(s|ed|ing)?\b|fail|died|dead|stopped|corrupt|defective|reliab|error|crash|broke|lost|issue|problem",
       "compatibility": r"compatib|recogni[sz]|detect|format|adapter|fits?\b|works? with|slot|support",
       "video & photo": r"video|record|1080|4k|photo|picture|\bhd\b|footage|shoot",
       "delivery": r"ship|deliver|arriv|packag|seller|received|box",
       "authenticity": r"fake|counterfeit|genuine|authentic|legit|real sandisk|knock"}
def clauses(t):
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", t):
        out += re.split(r",?\s+(?:but|however|although|though|except|yet)\s+", s)
    return [c.strip() for c in out if len(c.split()) >= 2]
rows = []
for r in df.itertuples():
    for c in clauses(r.review_text):
        hit = [(a, m.group(0)) for a, p in ASP.items() if (m := re.search(p, c, re.I))]
        rows.append(dict(review_id=r.review_id, clause=c, aspects=hit))
cl = pd.DataFrame(rows)

# ---------- 3. sentiment: stars -> clause classifier, out-of-fold ----------
cl = cl.merge(df[["review_id", "rating", "segment", "quarter"]], on="review_id")
cl["y"] = np.where(cl.rating >= 4, 1, np.where(cl.rating <= 2, 0, -1))
cl["p"] = np.nan
folds = KFold(5, shuffle=True, random_state=7)
ids = df.review_id.values
for tr, te in folds.split(ids):
    trm, tem = cl.review_id.isin(ids[tr]) & (cl.y >= 0), cl.review_id.isin(ids[te])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(vec.fit_transform(cl.clause[trm]), cl.y[trm])
    cl.loc[tem, "p"] = clf.predict_proba(vec.transform(cl.clause[tem]))[:, 1]
cl["sent"] = np.where(cl.p >= .65, "positive", np.where(cl.p <= .35, "negative", "neutral"))
m = cl[cl.aspects.str.len() > 0].explode("aspects").reset_index(drop=True)
m["aspect"] = m.aspects.str[0]; m["keyword"] = m.aspects.str[1].str.lower()
m = m.drop(columns="aspects")
m["quarter"] = m.quarter.astype(str)
m[["review_id", "aspect", "keyword", "clause", "sent", "p", "segment", "quarter", "rating"]].round(3) \
    .rename(columns={"sent": "sentiment", "p": "prob_positive"}).to_csv(f"{OUT}/aspect_mentions.csv", index=False)
ev = m[m.rating != 3]; yt = np.where(ev.rating >= 4, "positive", "negative"); yp = ev.sent.replace("neutral", "positive")
print("mentions:", len(m), "| distribution:", m.sent.value_counts(normalize=True).round(2).to_dict())
print("proxy check vs stars (out-of-fold, binary, neutral->pos): acc=%.3f macroF1=%.3f" %
      (accuracy_score(yt, yp), f1_score(yt, yp, average="macro")))
if os.path.exists(f"{OUT}/to_label_sample.csv"):
    print("to_label_sample.csv already exists, left untouched (your labels are safe)")
else:
    m.sample(300, random_state=3)[["review_id", "aspect", "clause"]].assign(true_sentiment="") \
        .to_csv(f"{OUT}/to_label_sample.csv", index=False)
