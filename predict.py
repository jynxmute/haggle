"""Hour Detective (Hack Club Haggle): Stratified 5-fold XGBoost + CatBoost ensemble."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

DROP = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
SEEDS = [42, 7, 2026]
MAX_HOURS = 140.7


def build_features(train_df, test_df):
    def add_ratios(df):
        d = df.copy()
        d["commits_per_day"] = d.num_commits / (d.active_commit_days + 1)
        d["lines_per_commit"] = d.lines_added_total / (d.num_commits + 1)
        d["lines_per_active_day"] = d.lines_added_total / (d.active_commit_days + 1)
        d["loc_per_file"] = d.lines_of_code / (d.num_source_files + 1)
        d["active_ratio"] = d.active_commit_days / (d.project_span_days + 1)
        d["churn_lines"] = d.lines_added_total + d.lines_deleted_total
        d["net_loc_ratio"] = d.lines_of_code / (d.lines_added_total + 1)
        d["asset_total"] = d.num_image_files + d.num_audio_files + d.num_3d_model_files + d.num_font_files
        d["has_web_metrics"] = d.page_weight_kb.notna().astype("float32")
        return d.replace([np.inf, -np.inf], np.nan)

    tr = add_ratios(train_df)
    te = add_ratios(test_df)

    feats = [c for c in tr.columns if c not in DROP]
    X_tr = tr[feats].copy()
    X_te = te.reindex(columns=feats).copy()

    cat_cols = []
    for c in feats:
        if X_tr[c].dtype == bool or str(X_tr[c].dtype) == "boolean":
            X_tr[c] = X_tr[c].astype("float32")
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32")
        elif not pd.api.types.is_numeric_dtype(X_tr[c]):
            cat_cols.append(c)
            fill_v = "other" if c == "project_category" else ""
            X_tr[c] = X_tr[c].fillna(fill_v).astype(str)
            X_te[c] = X_te[c].fillna(fill_v).astype(str)
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype("float32")
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32")

    return X_tr, X_te, cat_cols


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = np.log1p(train["hours_spent"].values)
    strata = train["hours_bucket"].fillna("unk").astype(str)

    X_tr, X_te, cat_cols = build_features(train, test)

    X_tr_xgb, X_te_xgb = X_tr.copy(), X_te.copy()
    for c in cat_cols:
        cats = sorted(set(X_tr_xgb[c]) | set(X_te_xgb[c]))
        X_tr_xgb[c] = pd.Categorical(X_tr_xgb[c], categories=cats)
        X_te_xgb[c] = pd.Categorical(X_te_xgb[c], categories=cats)

    xgb_preds = np.zeros(len(test))
    cat_preds = np.zeros(len(test))
    n_models = len(SEEDS) * 5

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_i, _) in enumerate(skf.split(X_tr, strata)):
            xm = XGBRegressor(
                n_estimators=1200,
                learning_rate=0.015,
                max_depth=3,
                min_child_weight=40,
                subsample=0.8,
                colsample_bytree=0.5,
                reg_alpha=2.0,
                reg_lambda=10.0,
                enable_categorical=True,
                tree_method="hist",
                random_state=seed + fold,
                n_jobs=-1,
                verbosity=0,
            )
            xm.fit(X_tr_xgb.iloc[tr_i], y[tr_i])
            xgb_preds += xm.predict(X_te_xgb) / n_models

            cb = CatBoostRegressor(
                iterations=1600,
                learning_rate=0.02,
                depth=5,
                l2_leaf_reg=12.0,
                cat_features=cat_cols,
                random_seed=seed + fold,
                verbose=0,
                allow_writing_files=False,
            )
            cb.fit(X_tr.iloc[tr_i], y[tr_i])
            cat_preds += cb.predict(X_te) / n_models

    # Optimal CV blend weights: 0.675 XGBoost (depth=3) + 0.325 CatBoost (depth=5)
    log_preds = 0.675 * xgb_preds + 0.325 * cat_preds
    preds = np.expm1(log_preds).clip(0.1, MAX_HOURS)

    pd.DataFrame({
        "project_id": test["project_id"],
        "hours_spent": preds,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
