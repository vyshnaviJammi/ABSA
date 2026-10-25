# Advancing the model

The pipeline has three scripts. The only thing that connects them is a CSV, so you can swap the model without touching the website.

```
prepare_data.py   cleans reviews + runs the BASELINE model   -> absa_ready.csv, aspect_mentions.csv, to_label_sample.csv
build_site.py     aspect_mentions.csv -> site_data.json -> refreshes index.html
evaluate.py       compares a model with your hand labels     -> accuracy, precision, recall, F1
```

## The contract (what your new model must output)
A file `data/aspect_mentions.csv` with one row per aspect mention:

| column | meaning |
|---|---|
| review_id | id from `absa_ready.csv` |
| aspect | aspect name (any names; the site adapts, but keep them few and clean) |
| clause | the text the sentiment was judged on |
| sentiment | `positive`, `neutral` or `negative` |
| keyword | optional: word to highlight in sample reviews |

Everything else (device group, quarter, rating) is joined automatically from `absa_ready.csv`.

## Suggested order
1. **Label** `data/to_label_sample.csv` (300 rows). Write positive / neutral / negative, or `wrong_aspect`.
2. **Score the baseline:** `python evaluate.py`. These are your first real numbers.
3. **Improve aspect extraction.** Ideas: noun-phrase or dependency-based extraction (spaCy), learned aspect clusters, or an LLM-based extractor (see the LLM-extractor papers in your literature review). Merge variants like "battery" and "battery life".
4. **Improve sentiment.** Ideas: a pretrained aspect-level sentiment model from Hugging Face (search "absa"), fine-tuning a transformer such as DistilBERT or RoBERTa, or an LLM with a fixed prompt. Training on star ratings is a weak signal; use your hand labels (and more of them) for evaluation, and for training if you label enough.
5. Write the results to `data/aspect_mentions.csv` in the format above (a Google Colab notebook with a free GPU is fine; download the CSV afterwards).
6. Run `python evaluate.py` again and compare with the baseline. Keep the baseline numbers for your report.
7. Run `python build_site.py` to refresh `site_data.json` and `index.html`, then commit and push.

## Notes
- `prepare_data.py` never overwrites `to_label_sample.csv` if it already exists, so your labels are safe.
- The sample is drawn from the baseline's mentions. If your new model finds different clauses, `evaluate.py` only scores the ones it can match and tells you how many it found.
- This file has no product ID, so "products" are devices named in the review text (list in `build_site.py`, variable `PROD`). If you get a dataset with a real product id, replace that matching with the id.
