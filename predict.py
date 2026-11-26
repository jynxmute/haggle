"""Hour Detective (Hack Club Haggle): 10-Fold Stacked Ensemble."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def main():
    test = pd.read_csv("test.csv")
    mdir = Path("models_v2")
    cfg = json.loads((mdir / "config.json").read_text())

    keep_feats = cfg["keep_feats"]
    cat_cols = cfg["cat_cols"]
    bool_cols = cfg["bool_cols"]
    num_cols = cfg["num_cols"]

    X_cb = test.reindex(columns=keep_feats).copy()
    for c in bool_cols + num_cols:
        X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce").astype("float32")
    for c in cat_cols:
        fill_v = "other" if c == "project_category" else ""
        X_cb[c] = X_cb[c].fillna(fill_v).astype(str)

    X_lgb = X_cb.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")

    p1 = np.zeros(len(test))
    p2 = np.zeros(len(test))
    p3 = np.zeros(len(test))
    p4 = np.zeros(len(test))

    for fold in range(10):
        m1 = CatBoostRegressor()
        m1.load_model(str(mdir / f"cb_rmse_{fold}.cbm"))
        p1 += m1.predict(X_cb) / 10.0

        m2 = CatBoostRegressor()
        m2.load_model(str(mdir / f"cb_huber_{fold}.cbm"))
        p2 += m2.predict(X_cb) / 10.0

        m3 = joblib.load(mdir / f"lgb_{fold}.pkl")
        p3 += m3.predict(X_lgb) / 10.0

        m4 = joblib.load(mdir / f"mlp_{fold}.pkl")
        p4 += m4.predict(X_cb) / 10.0

    meta = joblib.load(mdir / "meta_stacker.pkl")
    log_pred = meta.predict(np.column_stack([p1, p2, p3, p4]))
    preds = np.maximum(0.05, np.expm1(log_pred))

    pd.DataFrame({
        "project_id": test["project_id"],
        "hours_spent": preds,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
