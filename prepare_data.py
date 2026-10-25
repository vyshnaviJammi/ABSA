"""DSCI-12: clean the Amazon reviews CSV and build a baseline ABSA output.
Usage: python prepare_data.py amazon_reviews_cleaned.csv
Outputs: absa_ready.csv, aspect_mentions.csv, to_label_sample.csv, site_data.json
Baseline only: lexicon aspects + weak-label sentiment (stars -> clause classifier)."""
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
m.sample(300, random_state=3)[["review_id", "aspect", "clause"]].assign(true_sentiment="") \
    .to_csv(f"{OUT}/to_label_sample.csv", index=False)

# ---------- 4. site json ----------
top = df.segment.value_counts().head(5)
segs = list(top.index)
def split(g):
    n = len(g); return dict(n=n, **{k: round(100 * (g.sent == k).mean()) for k in ["positive", "neutral", "negative"]}) if n else dict(n=0)
rng = np.random.RandomState(5); data = dict(
    n_reviews=len(df), n_with_device=int(df.segment.notna().sum()), date_min=str(df.date.min().date()), date_max=str(df.date.max().date()),
    segments=[dict(name=s, n=int(c)) for s, c in top.items()], aspects={})
qs = sorted(df.quarter.unique())
for a, g in m.groupby("aspect"):
    n = len(g); d = dict(n=n, pos=round(100 * (g.sent == "positive").mean()), neg=round(100 * (g.sent == "negative").mean()))
    d["neu"] = 100 - d["pos"] - d["neg"]
    d["seg"] = {s: split(g[g.segment == s]) for s in segs}
    d["trend"] = [(round(100 * (gq.sent == "positive").mean()) if len(gq) >= 8 else None) for q in qs for gq in [g[g.quarter == q]]]
    smp = []
    for s, k in [("positive", 2), ("negative", 2), ("neutral", 1)]:
        c = g[(g.sent == s) & g.clause.str.len().between(35, 220)]
        for r in c.iloc[rng.permutation(len(c))[:k]].itertuples(): smp.append(dict(s=s, seg=r.segment or "", t=r.clause, kw=r.keyword))
    d["samples"] = smp; data["aspects"][a] = d
PROD = {"Motorola phone": (["motorola", "moto", "razr"], r"motorola|moto x|moto g|razr|droid x|droid bionic|droid 4|atrix|xoom"),
        "Samsung Galaxy phone": (["samsung", "galaxy"], r"samsung|galaxy"),
        "HTC phone": (["htc"], r"\bhtc\b"), "LG phone": (["lg"], r"\blg\b"), "Nokia phone": (["nokia"], r"nokia"),
        "GoPro camera": (["gopro", "go pro"], r"gopro|go pro"), "Canon camera": (["canon"], r"canon"),
        "Sony camera": (["sony"], r"\bsony\b"), "Nook tablet": (["nook"], r"nook")}
low = df.review_text.str.lower(); data["products"] = {}
for name, (alias, pat) in PROD.items():   # product = reviews that name it (file has no product id)
    ids = set(df.review_id[low.str.contains(pat)])
    if len(ids) < 25: continue
    g = m[m.review_id.isin(ids)]
    d = dict(alias=alias, n_reviews=len(ids), n_mentions=len(g), avg_rating=round(float(df.rating[df.review_id.isin(ids)].mean()), 2),
             pos=round(100 * (g.sent == "positive").mean()), neg=round(100 * (g.sent == "negative").mean()), aspects={}, samples=[])
    d["neu"] = 100 - d["pos"] - d["neg"]
    for a, ga in g.groupby("aspect"):
        if len(ga) >= 5: d["aspects"][a] = dict(n=len(ga), **{k: round(100 * (ga.sent == s).mean()) for k, s in [("pos", "positive"), ("neu", "neutral"), ("neg", "negative")]})
    for s, k in [("positive", 2), ("negative", 2), ("neutral", 1)]:
        c = g[(g.sent == s) & g.clause.str.len().between(35, 220)]
        for r in c.iloc[rng.permutation(len(c))[:k]].itertuples(): d["samples"].append(dict(s=s, a=r.aspect, t=r.clause, kw=r.keyword))
    data["products"][name] = d
data["quarters"] = qs
json.dump(data, open(f"{OUT}/site_data.json", "w"), separators=(",", ":"))
print("segments:", top.to_dict(), "| aspects:", {a: v["n"] for a, v in data["aspects"].items()})
