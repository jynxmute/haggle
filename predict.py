"""Hour Detective (Hack Club Haggle): Fast inference from pre-trained CatBoost + XGBoost models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


def main():
    test = pd.read_csv("test.csv")
    model_dir = Path("models")
    meta = json.loads((model_dir / "meta.json").read_text())

    features = meta["features"]
    cat_cols = meta["cat_cols"]
    bool_cols = meta["bool_cols"]
    xgb_cats = meta["xgb_cats"]
    n_models = meta["n_models"]
    w_cb7, w_cb8, w_xgb = meta["weights"]

    X_base = test.reindex(columns=features).copy()

    # CatBoost input
    X_cb = X_base.copy()
    for c in features:
        if c in bool_cols:
            X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce").astype("float32")
        elif c in cat_cols:
            fill_v = "other" if c == "project_category" else ""
            X_cb[c] = X_cb[c].fillna(fill_v).astype(str)
        else:
            X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce").astype("float32")

    # XGBoost input
    X_xgb = X_base.copy()
    for c in bool_cols:
        X_xgb[c] = X_xgb[c].astype("boolean")
    for c in cat_cols:
        X_xgb[c] = pd.Categorical(X_xgb[c], categories=xgb_cats[c])
    for c in features:
        if c not in bool_cols and c not in cat_cols:
            X_xgb[c] = pd.to_numeric(X_xgb[c], errors="coerce").astype("float32")

    p_cb7 = np.zeros(len(test))
    p_cb8 = np.zeros(len(test))
    p_xgb = np.zeros(len(test))

    for i in range(n_models):
        m7 = CatBoostRegressor()
        m7.load_model(str(model_dir / f"cb7_{i}.cbm"))
        p_cb7 += m7.predict(X_cb) / n_models

        m8 = CatBoostRegressor()
        m8.load_model(str(model_dir / f"cb8_{i}.cbm"))
        p_cb8 += m8.predict(X_cb) / n_models

        mx = XGBRegressor()
        mx.load_model(str(model_dir / f"xgb_{i}.json"))
        p_xgb += mx.predict(X_xgb) / n_models

    log_pred = w_cb7 * p_cb7 + w_cb8 * p_cb8 + w_xgb * p_xgb
    preds = np.maximum(0.05, np.expm1(log_pred))

    pd.DataFrame({
        "project_id": test["project_id"],
        "hours_spent": preds,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
