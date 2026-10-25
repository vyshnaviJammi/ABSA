# DSCI-12: Aspect-Based Sentiment and Voice-of-Customer Analytics

Baseline ABSA pipeline and dashboard on about 4,900 Amazon reviews of SanDisk memory cards (2012-2014).

## What is in this repo
- `index.html`: the dashboard (top 5 devices pie chart, aspect search, per-aspect dashboard)
- `prepare_data.py`: cleans the reviews, extracts aspects, predicts sentiment, writes the outputs
- `data/amazon_reviews_cleaned.csv`: input reviews
- `data/absa_ready.csv`: cleaned reviews
- `data/aspect_mentions.csv`: one row per aspect mention with predicted sentiment
- `data/to_label_sample.csv`: 300 mentions for manual labelling (validation)
- `data/site_data.json`: summary numbers used by the dashboard

## Run the pipeline
```
pip install -r requirements.txt
python prepare_data.py data/amazon_reviews_cleaned.csv
```
Outputs are written into the `data/` folder.

## View the dashboard
Open `index.html` in a browser (internet needed to load Chart.js), or visit the live site: <add your GitHub Pages link>

## Method (baseline)
- Aspects: keyword lists for speed, storage, price, reliability, compatibility, video and photo, delivery, authenticity
- Sentiment: logistic regression trained on star ratings, applied to the clause containing the aspect (5-fold out-of-fold predictions)
- Note: the dataset has no product ID, so reviews are grouped by the first device mentioned
- Status: baseline only; to be replaced by the final ABSA model and validated on hand-labelled data
