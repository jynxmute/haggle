"""Hour Detective (Hack Club Haggle, Week 1): predict hours_spent for a project.

Reads train.csv and test.csv from the current folder, trains a blend of tree
models on log(1 + hours) and writes predictions.csv (project_id, hours_spent).
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

T0 = time.time()
TIME_BUDGET = 300          # seconds. The grader allows 10 minutes per run.
SEEDS = [42, 7, 2026]      # each model is averaged over these seeds
MAX_HOURS = 140.7          # largest value in train.csv
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
    return d.replace([np.inf, -np.inf], np.nan)


train = add_features(train)
test = add_features(test)
features = [c for c in train.columns if c not in DROP]
for c in features:
    if c not in test.columns:
        test[c] = np.nan

# Text and True/False columns become categories with the SAME categories in
# train and test. Numbers stay numbers (missing values are fine for the models).
cat_cols = []
for c in features:
    if train[c].dtype == bool or not pd.api.types.is_numeric_dtype(train[c]):
        cat_cols.append(c)
        tr_s = train[c].astype(object).where(train[c].notna(), "MISSING").astype(str)
        te_s = test[c].astype(object).where(test[c].notna(), "MISSING").astype(str)
        cats = sorted(set(tr_s) | set(te_s))
        train[c] = pd.Categorical(tr_s, categories=cats)
        test[c] = pd.Categorical(te_s, categories=cats)
    else:
        test[c] = pd.to_numeric(test[c], errors="coerce")

y = np.log1p(train["hours_spent"].values)
Xtr, Xte = train[features], test[features]

# One-hot version for ExtraTrees
both = pd.concat([Xtr, Xte], keys=["tr", "te"])
both = pd.get_dummies(both, columns=cat_cols, dtype=float).fillna(-1)
Etr, Ete = both.loc["tr"], both.loc["te"]


def hgb(seed):
    m = HistGradientBoostingRegressor(
        max_iter=900, learning_rate=0.015, max_leaf_nodes=6,
        min_samples_leaf=60, l2_regularization=10.0, max_features=0.5,
        categorical_features="from_dtype", random_state=seed,
    )
    return m.fit(Xtr, y).predict(Xte)


def lgbm(seed):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.015, num_leaves=7,
        min_child_samples=55, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=10.0,
        random_state=seed, verbose=-1,
    )
    return m.fit(Xtr, y).predict(Xte)


def cat(seed):
    from catboost import CatBoostRegressor
    a, b = Xtr.copy(), Xte.copy()
    for c in cat_cols:
        a[c] = a[c].astype(str)
        b[c] = b[c].astype(str)
    m = CatBoostRegressor(
        iterations=1400, learning_rate=0.02, depth=4, l2_leaf_reg=10.0,
        cat_features=cat_cols, random_seed=seed, verbose=0,
        allow_writing_files=False,
    )
    return m.fit(a, y).predict(b)


def extra_trees(seed):
    m = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=0.5,
        n_jobs=-1, random_state=seed,
    )
    return m.fit(Etr, y).predict(Ete)


# Each model is wrapped so one failure (or a missing library) never crashes
# the whole run. Extra seeds are skipped if we are running out of time.
parts = {}
for name, fn in [("hgb", hgb), ("lgb", lgbm), ("cat", cat), ("et", extra_trees)]:
    out = []
    for seed in SEEDS:
        if out and time.time() - T0 > TIME_BUDGET:
            break
        try:
            out.append(fn(seed))
        except Exception as e:
            print(f"{name} skipped (seed {seed}): {e}")
            break
    if out:
        parts[name] = np.mean(out, axis=0)
        print(f"{name}: done ({len(out)} seeds, {time.time() - T0:.0f}s)")

if parts:
    log_pred = np.mean(list(parts.values()), axis=0)   # equal blend in log space
else:
    log_pred = np.full(len(test), y.mean())            # last-resort fallback

preds = np.expm1(log_pred).clip(0.1, MAX_HOURS)

sub = pd.DataFrame({
    "project_id": test["project_id"],
    "hours_spent": np.round(preds, 2),
})
sub.to_csv("predictions.csv", index=False)
print(f"wrote predictions.csv with {len(sub)} rows")
