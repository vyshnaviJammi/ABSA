"""DSCI-12 step 2: build the website data from aspect_mentions.csv (baseline OR your own model).
Usage: python build_site.py [data_folder] [index.html]     (defaults: data  index.html)
Input contract for aspect_mentions.csv: review_id, aspect, clause, sentiment (positive/neutral/negative);
optional: keyword (word to highlight). Everything else is joined from absa_ready.csv."""
import re, sys, os, json, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "data"
HTML = sys.argv[2] if len(sys.argv) > 2 else "index.html"
df = pd.read_csv(f"{D}/absa_ready.csv", parse_dates=["date"])
m = pd.read_csv(f"{D}/aspect_mentions.csv")
m = m.drop(columns=[c for c in ["segment", "quarter", "rating"] if c in m.columns]) \
     .merge(df[["review_id", "segment", "quarter", "rating"]], on="review_id", how="left").rename(columns={"sentiment": "sent"})
m["sent"] = m["sent"].str.lower().str.strip()
m = m[m.sent.isin(["positive", "neutral", "negative"])]
m["keyword"] = (m["keyword"] if "keyword" in m.columns else m["aspect"]).fillna(m["aspect"]).astype(str).str.lower()

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
        for r in c.iloc[rng.permutation(len(c))[:k]].itertuples(): smp.append(dict(s=s, seg=r.segment if isinstance(r.segment, str) else "", t=r.clause, kw=r.keyword))
    d["samples"] = smp; data["aspects"][a] = d
PROD = {"Samsung Galaxy S3": (["galaxy s3", "s3", "siii"], r"\bs3\b|\bsiii\b|\bgs3\b|\bs iii\b|galaxy s3"),
        "Samsung Galaxy S4": (["galaxy s4", "s4"], r"\bs4\b|galaxy s4|\bs iv\b"),
        "Samsung Galaxy Note": (["galaxy note", "note"], r"galaxy note|note ?[23]\b|note ii|note 10\.1|samsung note"),
        "Samsung Galaxy Tab": (["galaxy tab", "tab"], r"galaxy tab|\btab ?[234]\b|samsung tab"),
        "Samsung Galaxy phone": (["samsung", "galaxy"], r"samsung|galaxy"),
        "Motorola phone": (["motorola", "moto", "razr"], r"motorola|moto x|moto g|razr|droid x|droid bionic|droid 4|atrix|xoom"),
        "HTC phone": (["htc"], r"\bhtc\b"), "LG phone": (["lg"], r"\blg\b"),
        "Nokia Lumia phone": (["nokia", "lumia"], r"nokia|lumia"), "BlackBerry phone": (["blackberry"], r"blackberry"),
        "Apple iPhone": (["iphone"], r"iphone"), "Apple MacBook": (["macbook", "mac"], r"macbook"),
        "GoPro camera": (["gopro", "go pro", "hero"], r"gopro|go pro|hero ?[34]"), "Canon camera": (["canon"], r"canon"),
        "Sony camera": (["sony"], r"\bsony\b"), "Nikon camera": (["nikon"], r"nikon"), "DSLR camera": (["dslr"], r"dslr"),
        "Dash cam": (["dash cam", "dashcam"], r"dash ?cam"), "Garmin GPS": (["garmin", "gps"], r"garmin"),
        "Nook tablet": (["nook"], r"nook"), "Asus tablet": (["asus", "transformer"], r"asus|transformer|nexus 7"),
        "Microsoft Surface": (["surface"], r"surface (pro|rt|2|3|tablet)|microsoft surface"), "Dell device": (["dell", "venue"], r"\bdell\b|venue")}
low = df.review_text.str.lower(); data["products"] = {}
for name, (alias, pat) in PROD.items():   # product = reviews that name it (file has no product id)
    ids = set(df.review_id[low.str.contains(pat)])
    if len(ids) < 15: continue
    g = m[m.review_id.isin(ids)]
    d = dict(alias=alias, n_reviews=len(ids), n_mentions=len(g), avg_rating=round(float(df.rating[df.review_id.isin(ids)].mean()), 2),
             pos=round(100 * (g.sent == "positive").mean()), neg=round(100 * (g.sent == "negative").mean()), aspects={}, samples=[])
    d["neu"] = 100 - d["pos"] - d["neg"]
    d["trend"] = [(round(100 * (gq.sent == "positive").mean()) if len(gq) >= 8 else None) for q in qs for gq in [g[g.quarter == q]]]
    for a, ga in g.groupby("aspect"):
        if len(ga) >= 5: d["aspects"][a] = dict(n=len(ga), **{k: round(100 * (ga.sent == s).mean()) for k, s in [("pos", "positive"), ("neu", "neutral"), ("neg", "negative")]})
    for s, k in [("positive", 2), ("negative", 2), ("neutral", 1)]:
        c = g[(g.sent == s) & g.clause.str.len().between(35, 220)]
        for r in c.iloc[rng.permutation(len(c))[:k]].itertuples(): d["samples"].append(dict(s=s, a=r.aspect, t=r.clause, kw=r.keyword))
    data["products"][name] = d
data["products"] = dict(sorted(data["products"].items(), key=lambda kv: -kv[1]["n_reviews"]))
data["quarters"] = qs
txt = json.dumps(data, separators=(",", ":"), allow_nan=False)
open(f"{D}/site_data.json", "w").write(txt)
if os.path.exists(HTML):
    h = open(HTML).read()
    new = re.sub(r"/\*DATA_START\*/.*?/\*DATA_END\*/", lambda _: "/*DATA_START*/" + txt.replace("</", "<\\/") + "/*DATA_END*/", h, flags=re.S)
    open(HTML, "w").write(new); print("updated", HTML)

print("segments:", top.to_dict(), "| aspects:", {a: v["n"] for a, v in data["aspects"].items()})
