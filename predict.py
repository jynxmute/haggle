import numpy as np
import pandas as pd
import lightbgm as lgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
features = [c for c in train.columns if c not in drop_cols]

for col in features:
    if train[col].dtype == "object":
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(train[features], train["log_hours"])

preds = np.expm1(model.predict(test[features])).clip(0.1, 140.7)

sub = pd.DataFrame({
    "project_id": test["project_id"]
    "hours_spent": np.round(preds, 2)
})
sub.to_csv("predictions.csv", index=False)