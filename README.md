# DSCI-12: Aspect-Based Sentiment and Voice-of-Customer Analytics

Baseline aspect-based sentiment analysis (ABSA) pipeline and interactive dashboard on about 4,900 Amazon reviews of SanDisk memory cards (2012-2014).

Live dashboard: <add your GitHub Pages link>

## Dashboard
- Home: top 5 device groups (pie chart), search box, buttons for popular products and aspects
- Search an **aspect** (speed, storage, price, reliability, compatibility, video & photo, delivery, authenticity): sentiment split, sentiment by device, trend by quarter, sample reviews
- Search a **product** (23 devices named in the reviews, e.g. "Motorola phone"): sentiment, sentiment by aspect, trend by quarter, sample reviews

## Repo layout
```
index.html                  dashboard (data embedded, open in any browser)
prepare_data.py             step 1: clean reviews + baseline ABSA model
build_site.py               step 2: aspect_mentions.csv -> site_data.json -> index.html
evaluate.py                 step 3: score a model against hand labels
docs/ADVANCING_THE_MODEL.md how to plug in a better model
data/amazon_reviews_cleaned.csv   input reviews
data/absa_ready.csv               cleaned reviews (+ device group, quarter, star label)
data/aspect_mentions.csv          one row per aspect mention with predicted sentiment
data/to_label_sample.csv          300 mentions to label by hand (validation)
data/site_data.json               summary numbers used by the dashboard
```

## Run
```
pip install -r requirements.txt
python prepare_data.py data/amazon_reviews_cleaned.csv
python build_site.py
python evaluate.py          # after filling data/to_label_sample.csv
```

## Method (baseline)
- Cleaning: removed duplicates and reviews under 3 words, fixed run-together sentences, dropped reviewer names
- Aspects: keyword lists for 8 aspects, applied clause by clause
- Sentiment: logistic regression on TF-IDF, trained on star ratings (weak labels), 5-fold out-of-fold predictions
- Check against stars (weak proxy only): accuracy 0.89, macro-F1 0.71. Proper evaluation needs the hand-labelled sample.

## Limitations
- The dataset has no product ID, so "products" are devices named in the review text and groups overlap
- Aspect extraction is rule-based and sentiment labels come from star ratings, so this is a baseline, not the final model
- Small products (under 30 reviews) have rough percentages

## Status
CP1 baseline. Next: hand-label the sample, evaluate, then replace the baseline with a stronger model (see docs/ADVANCING_THE_MODEL.md).
