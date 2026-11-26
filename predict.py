"""Hour Detective (Hack Club Haggle): 3-seed x 10-fold regularized CatBoost on raw features."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

DROP = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
SEEDS =


def prepare_data(train_df, test_df):
    features = [c for c in train_df.columns if c not in DROP]
    X_tr = train_df[features].copy()
    X_te = test_df.reindex(columns=features).copy()

    cat_cols = []
    for c in features:
        if X_tr[c].dtype == bool or str(X_tr[c].dtype) == "boolean":
            X_tr[c] = X_tr[c].astype("float32")
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32")
        elif not pd.api.types.is_numeric_dtype(X_tr[c]):
            cat_cols.append(c)
            fill_val = "other" if c == "project_category" else ""
            X_tr[c] = X_tr[c].fillna(fill_val).astype(str)
            X_te[c] = X_te[c].fillna(fill_val).astype(str)
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype("float32")
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32")

    return X_tr, X_te, cat_cols


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = np.log1p(train["hours_spent"].values)
    strata = train["hours_bucket"].fillna("unk").astype(str)

    X_tr, X_te, cat_cols = prepare_data(train, test)

    cat_preds = np.zeros(len(test))
    n_models = len(SEEDS) * 10

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for fold, (tr_i, _) in enumerate(skf.split(X_tr, strata)):
            cb = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.02,
                depth=5,
                l2_leaf_reg=10.0,
                random_strength=2.0,
                subsample=0.8,
                bootstrap_type="Bernoulli",
                cat_features=cat_cols,
                random_seed=seed + fold,
                verbose=0,
                allow_writing_files=False,
            )
            cb.fit(X_tr.iloc[tr_i], y[tr_i])
            cat_preds += cb.predict(X_te) / n_models

    preds = np.maximum(0.05, np.expm1(cat_preds))

    pd.DataFrame({
        "project_id": test["project_id"],
        "hours_spent": preds,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
