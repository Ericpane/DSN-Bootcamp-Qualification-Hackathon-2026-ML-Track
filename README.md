# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Predicting `total_sales` for product-store combinations. Evaluation metric: RMSE.

## Result

**5-fold CV RMSE: 1073.59** (CatBoost, engineered features, Optuna-tuned hyperparameters)

## How to run

```bash
pip install -r requirements.txt

# Place train.csv and test.csv inside data/
python src/pipeline.py --data-dir data --output-dir output
```

This prints per-fold and mean CV RMSE, then trains a 5-model bagged ensemble
(one per CV fold) and writes `output/submission.csv`.

## Approach

1. **Cleaning**: normalize inconsistent `fat_content` text variants and
   `product_category` casing (raw data mixes UPPERCASE/Title Case/lowercase
   for the same 16 real categories).
2. **Imputation**: `product_weight_kg` (~18% missing) filled per-product
   median, falling back to global median. `store_size` (~28% missing) filled
   per-store-code/format mode, falling back to global mode.
3. **Feature engineering**: price-per-kg, price×visibility interaction, log
   transforms, and store/product-level price & weight aggregates.
4. **Model**: CatBoostRegressor, hyperparameters tuned via a 40-trial Optuna
   search (see "What was tried" below).
5. **Validation**: 5-fold cross-validation with aggregate features
   recomputed *inside* each fold using only that fold's training rows — see
   "Leakage safety" below for why this matters.

## Leakage safety

The store/product aggregate features (e.g. "average price for this store")
are recomputed separately for each CV fold using only that fold's training
split — never the full training set and never validation rows. Computing
them once on the full dataset before splitting would let validation-row
information leak into features the model uses to predict those same
rows, silently inflating the reported CV score relative to real
held-out performance. This was checked directly: the leakage-safe version
of the pipeline scores 1074.79, essentially the same as the
leakage-prone version (1073.36) — this dataset's aggregates weren't leaking
meaningfully, but the check was necessary to confirm that rather than
assume it.

## What was tried and rejected

Everything below was tested with the same leakage-safe 5-fold CV, so the
numbers are directly comparable. Listed because a big part of doing this
honestly is documenting the dead ends, not just the thing that worked.

| Approach | CV RMSE | Verdict |
|---|---:|---|
| **CatBoost + engineered features + Optuna tuning (this repo)** | **1073.59** | best |
| CatBoost, no feature engineering at all | 1073.18 | statistically identical |
| Deep-cleaned data (store-level imputation, missingness flags) | 1075.57 | no improvement |
| KMeans clustering + PCA features | 1075.41 | no improvement |
| Log-transformed target (log1p, back-transformed) | 1108.0 | worse |
| Neural network (embeddings for categoricals) | 1175.10 | worse, less stable |
| Smoothed target encoding on `product_code`/`store_code` | 1180+ | worse — overfits (median ~4 rows/product) |
| LightGBM alone / CatBoost+LightGBM blend | 1181–1217 | worse |

**Conclusion**: multiple independent approaches — including a completely
separate from-scratch notebook someone else wrote, and a from-scratch
deep-cleaning pass — converge on ~1073-1076 CV RMSE. This appears to be
close to the practical floor for this dataset given the available features,
not a modeling or cleaning failure. Public leaderboard scores (as of
2 days before deadline) show the bulk of legitimate submissions clustered
in the same 1065-1076 range, which supports this conclusion externally.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── src/
│   └── pipeline.py       # cleaning, feature engineering, CV, training, prediction
├── data/                 # train.csv / test.csv (not committed — see .gitignore)
└── output/               # generated submission.csv (not committed)
```
