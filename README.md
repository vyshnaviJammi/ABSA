# DSCI-12: Aspect-Based Sentiment Analysis

This project analyzes around 4,900 Amazon reviews of SanDisk memory cards from 2012–2014 using Aspect-Based Sentiment Analysis (ABSA).

It identifies what customers discuss, such as speed, storage, and price, and predicts whether their opinions are positive, neutral, or negative.

## Dashboard

The interactive dashboard allows users to:
- View popular device groups.
- Search for aspects and products.
- Explore sentiment distributions and quarterly trends.
- Read sample customer reviews.

## Technologies

- Python, Pandas, NumPy
- PyTorch and Hugging Face Transformers
- DeBERTa ABSA model
- HTML, CSS, JavaScript, Chart.js

## Project Files

- `transformer_absa.py` — Cleans reviews, extracts aspects, and predicts sentiment.
- `prepare_data.py` — Creates baseline ABSA data.
- `build_site.py` — Generates dashboard data and updates `index.html`.
- `evaluate.py` — Evaluates predictions against manually labelled data.
- `index.html` — Interactive dashboard.
- `data/` — Input reviews and generated results.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Transformer ABSA pipeline:

```bash
python transformer_absa.py data/amazon_reviews_cleaned.csv
```

Build the dashboard:

```bash
python build_site.py
```

Open `index.html` in a browser.
