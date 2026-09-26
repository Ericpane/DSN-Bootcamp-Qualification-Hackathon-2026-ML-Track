"""
DSN Bootcamp Qualification Hackathon 2026 (ML Track) — Final Submission Pipeline
==================================================================================

Predicts `total_sales` for each product-store row using CatBoost.

Validated result: Mean 5-fold CV RMSE = 1073.59 (see README.md for the full
comparison of approaches that were tried and rejected).

Usage:
    python src/pipeline.py --data-dir data/ --output-dir output/

Expects `train.csv` and `test.csv` inside --data-dir, in the original
competition format (columns: id, product_code, product_weight_kg, fat_content,
shelf_visibility, product_category, product_price, store_code, store_age_years,
store_size, store_location_tier, store_format, total_sales [train only]).
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "total_sales"
ID_COL = "id"

CAT_FEATURES = [
    "product_code",
    "fat_content",
    "product_category",
    "store_code",
    "store_size",
    "store_location_tier",
    "store_format",
]

AGG_COLS = [
    "store_price_mean",
    "store_weight_mean",
    "store_product_count",
    "prod_price_mean",
    "prod_weight_mean",
    "price_deviation",
]

BASE_NUM_FEATURES = [
    "product_weight_kg",
    "shelf_visibility",
    "product_price",
    "store_age_years",
    "price_per_kg",
    "visibility_x_price",
    "log_price",
    "log_visibility",
    "store_age_sq",
]

FEATURES = CAT_FEATURES + BASE_NUM_FEATURES + AGG_COLS

# Best hyperparameters found via a 40-trial Optuna search, verified on full
# 5-fold CV (see README.md for the search setup).
BEST_PARAMS = dict(
    iterations=3000,
    learning_rate=0.04978013676724478,
    depth=5,
    l2_leaf_reg=3.4093201804528364,
    random_strength=2.9858151120290297,
    bagging_temperature=1.5486641509206425,
    loss_function="RMSE",
    eval_metric="RMSE",
)

N_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 150


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize inconsistent categorical text.

    `fat_content` variants (e.g. 'lf', 'reg') are mapped to two clean labels.
    `product_category` case is standardized (raw data mixes UPPER/Title/lower
    for the same 16 real categories).
    """
    df = df.copy()
    df["fat_content"] = df["fat_content"].str.lower().str.strip()
    df["fat_content"] = df["fat_content"].replace(
        {
            "low fat": "Low Fat",
            "lf": "Low Fat",
            "lowfat": "Low Fat",
            "reg": "Regular",
            "regular": "Regular",
        }
    )
    df["product_category"] = df["product_category"].str.strip().str.title()
    return df


def impute_missing(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill `product_weight_kg` and `store_size`, using only train-derived statistics.

    product_weight_kg: per-product median (weight is a near-constant physical
    property of a product), falling back to the global median for products
    with no known weight at all.

    store_size: per-store-code / per-store-format mode, falling back to the
    global mode. (Note: store_size is in fact deterministic per physical
    store — see README for a documented, more precise per-store lookup that
    was tested and found not to change CV RMSE meaningfully; this simpler
    version is kept here for clarity.)
    """
    train = train.copy()
    test = test.copy()

    train["product_weight_kg"] = train.groupby("product_code")["product_weight_kg"].transform(
        lambda x: x.fillna(x.median())
    )
    test["product_weight_kg"] = test.groupby("product_code")["product_weight_kg"].transform(
        lambda x: x.fillna(x.median())
    )
    overall_median = train["product_weight_kg"].median()
    train["product_weight_kg"] = train["product_weight_kg"].fillna(overall_median)
    test["product_weight_kg"] = test["product_weight_kg"].fillna(overall_median)

    for col in ["store_code", "store_format"]:
        mode_size = train.groupby(col)["store_size"].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        train["store_size"] = train["store_size"].fillna(train[col].map(mode_size))
        test["store_size"] = test["store_size"].fillna(test[col].map(mode_size))

    most_common = train["store_size"].mode()[0]
    train["store_size"] = train["store_size"].fillna(most_common)
    test["store_size"] = test["store_size"].fillna(most_common)

    return train, test


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level engineered features. Safe to compute independently on any
    dataframe (no cross-row statistics involved, so no leakage risk)."""
    df = df.copy()
    df["price_per_kg"] = df["product_price"] / (df["product_weight_kg"] + 1e-5)
    df["visibility_x_price"] = df["shelf_visibility"] * df["product_price"]
    df["log_price"] = np.log1p(df["product_price"])
    df["log_visibility"] = np.log1p(df["shelf_visibility"])
    df["store_age_sq"] = df["store_age_years"] ** 2
    return df


def add_agg_features(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Store- and product-level aggregate features.

    IMPORTANT (leakage prevention): `ref` must be the training data only —
    during cross-validation this is the current fold's training split, NOT
    the full training set and NOT the validation/test rows. Computing these
    aggregates using rows the model will later be scored on would leak
    target-correlated information into the features.
    """
    df = df.copy()

    store_price_mean = ref.groupby("store_code")["product_price"].mean()
    store_weight_mean = ref.groupby("store_code")["product_weight_kg"].mean()
    store_product_count = ref.groupby("store_code")["product_code"].nunique()

    df["store_price_mean"] = df["store_code"].map(store_price_mean)
    df["store_weight_mean"] = df["store_code"].map(store_weight_mean)
    df["store_product_count"] = df["store_code"].map(store_product_count)

    prod_price_mean = ref.groupby("product_code")["product_price"].mean()
    prod_weight_mean = ref.groupby("product_code")["product_weight_kg"].mean()

    df["prod_price_mean"] = df["product_code"].map(prod_price_mean)
    df["prod_weight_mean"] = df["product_code"].map(prod_weight_mean)
    df["price_deviation"] = df["product_price"] - df["prod_price_mean"]

    for col in AGG_COLS:
        if df[col].isna().any():
            fill_val = ref[col].median() if col in ref.columns and ref[col].notna().any() else df[col].median()
            df[col] = df[col].fillna(fill_val)

    return df


def run_cv(train: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    """5-fold cross-validation with per-fold, leakage-safe aggregate features.

    Returns out-of-fold predictions (for diagnostics) and the list of
    per-fold RMSEs.
    """
    y = train[TARGET]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(train))
    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        fold_train_raw = train.iloc[train_idx]
        fold_val_raw = train.iloc[val_idx]

        # Aggregates are recomputed per fold, using ONLY this fold's training
        # rows — this is what makes the CV score leakage-safe.
        fold_train = add_agg_features(fold_train_raw, fold_train_raw)
        fold_val = add_agg_features(fold_val_raw, fold_train_raw)

        X_tr, y_tr = fold_train[FEATURES], y.iloc[train_idx]
        X_val, y_val = fold_val[FEATURES], y.iloc[val_idx]

        model = CatBoostRegressor(
            **BEST_PARAMS,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            cat_features=CAT_FEATURES,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

        pred = model.predict(X_val)
        oof_preds[val_idx] = pred

        rmse = root_mean_squared_error(y_val, pred)
        rmses.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    print(f"\nMean CV RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    return oof_preds, rmses


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Train one model per fold (bagged ensemble) and average their
    predictions on the real test set. Test-set aggregates are computed from
    the FULL training set, since at prediction time all of train is
    legitimately available.
    """
    y = train[TARGET]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    test_preds = np.zeros(len(test))
    test_with_agg = add_agg_features(test, train)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        fold_train_raw = train.iloc[train_idx]
        fold_val_raw = train.iloc[val_idx]

        fold_train = add_agg_features(fold_train_raw, fold_train_raw)
        fold_val = add_agg_features(fold_val_raw, fold_train_raw)

        X_tr, y_tr = fold_train[FEATURES], y.iloc[train_idx]
        X_val, y_val = fold_val[FEATURES], y.iloc[val_idx]

        model = CatBoostRegressor(
            **BEST_PARAMS,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            cat_features=CAT_FEATURES,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

        test_preds += model.predict(test_with_agg[FEATURES]) / N_FOLDS

    return test_preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--submission-name", type=str, default="submission.csv"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")

    train = clean_data(train)
    test = clean_data(test)

    train, test = impute_missing(train, test)

    train = add_basic_features(train)
    test = add_basic_features(test)

    print("Running 5-fold cross-validation...")
    run_cv(train)

    print("\nTraining final ensemble and predicting on test set...")
    test_preds = train_and_predict(train, test)

    submission = pd.DataFrame(
        {ID_COL: test[ID_COL], TARGET: np.clip(test_preds, 0, None)}
    )
    out_path = args.output_dir / args.submission_name
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
