import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

DROP = ["project_id", "hours_spent", "log_hours", "hours_bucket"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def add_features(d):
    d = d.copy()
    d["commits_per_day"] = d.num_commits / (d.active_commit_days + 1)
    d["lines_per_commit"] = d.lines_added_total / (d.num_commits + 1)
    d["lines_per_active_day"] = d.lines_added_total / (d.active_commit_days + 1)
    d["loc_per_file"] = d.lines_of_code / (d.num_source_files + 1)
    d["log_lines_added"] = np.log1p(d.lines_added_total)
    d["log_commits"] = np.log1p(d.num_commits)
    d["log_loc"] = np.log1p(d.lines_of_code)
    d["span_per_active"] = d.project_span_days / (d.active_commit_days + 1)
    d["active_ratio"] = d.active_commit_days / (d.project_span_days + 1)
    d["asset_total"] = (d.num_image_files + d.num_audio_files
                        + d.num_3d_model_files + d.num_font_files)
    d["churn_lines"] = d.lines_added_total + d.lines_deleted_total
    d["net_loc_ratio"] = d.lines_of_code / (d.lines_added_total + 1)
    return d


train = add_features(train)
test = add_features(test)
features = [c for c in train.columns if c not in DROP]

cat_cols = []
for c in features:
    if train[c].dtype == bool:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
    elif not pd.api.types.is_numeric_dtype(train[c]):
        cat_cols.append(c)
        tr_s = train[c].fillna("MISSING").astype(str)
        te_s = test[c].fillna("MISSING").astype(str)
        cats = sorted(set(tr_s.unique()) | set(te_s.unique()))
        train[c] = pd.Categorical(tr_s, categories=cats)
        test[c] = pd.Categorical(te_s, categories=cats)

y = train["log_hours"].values

# CatBoost needs string/int categories
train_cb = train[features].copy()
test_cb = test[features].copy()
for c in cat_cols:
    train_cb[c] = train_cb[c].astype(str)
    test_cb[c] = test_cb[c].astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
p_hgb = np.zeros(len(test))
p_lgb = np.zeros(len(test))
p_cat = np.zeros(len(test))

for tr_idx, va_idx in kf.split(train):
    X_tr, y_tr = train.iloc[tr_idx][features], y[tr_idx]

    hgb = HistGradientBoostingRegressor(
        max_iter=900, learning_rate=0.015, max_leaf_nodes=6,
        min_samples_leaf=60, l2_regularization=10.0, max_features=0.5,
        categorical_features="from_dtype", random_state=42,
    )
    hgb.fit(X_tr, y_tr)
    p_hgb += hgb.predict(test[features]) / 5

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.015, num_leaves=7,
        min_child_samples=55, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=10.0,
        random_state=42, verbose=-1,
    )
    model_lgb.fit(X_tr, y_tr)
    p_lgb += model_lgb.predict(test[features]) / 5

    cb = CatBoostRegressor(
        iterations=1400, learning_rate=0.02, depth=4,
        l2_leaf_reg=10.0, cat_features=cat_cols,
        random_seed=42, verbose=0,
    )
    cb.fit(train_cb.iloc[tr_idx], y_tr)
    p_cat += cb.predict(test_cb) / 5

log_preds = 0.30 * p_hgb + 0.35 * p_lgb + 0.35 * p_cat
preds = np.maximum(0.05, np.expm1(log_preds))

sub = pd.DataFrame({
    "project_id": test["project_id"],
    "hours_spent": np.round(preds, 2),
})
sub.to_csv("predictions.csv", index=False)
