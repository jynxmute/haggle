# Hour Detective: predicting coding hours (Hack Club Haggle, Week 1)

This model predicts how many hours a project took (`hours_spent`) from stats about its GitHub repo: commits, lines of code, assets, README, hosting and more. It is scored with RMSLE (lower is better).

## How it works

- **Target:** the models learn `log(1 + hours)`. RMSLE is an error on the log scale, and hours are very skewed (median about 6, max 140.7).
- **Features:** all the columns in `train.csv` (except the ID and the answer columns) plus 12 simple ratios, such as commits per active day, lines per commit and total asset files.
- **Models:** four tree models, each averaged over 3 random seeds: HistGradientBoosting, LightGBM, CatBoost and ExtraTrees. All are heavily regularized (small trees, strong L2) because the labels are noisy.
- **Blend:** an equal average in log space, converted back with `expm1` and clipped to 0.1 to 140.7 hours, the range seen in `train.csv`.
- **Safety:** each model is wrapped so that if a library is missing or fails, the others still run and `predict.py` always writes a valid `predictions.csv`. It also stops adding seeds after 5 minutes to stay under the 10-minute limit.

## Results

All numbers are RMSLE on `train.csv`, using 5-fold cross-validation in `main.ipynb`:

| Model | CV RMSLE |
|---|---|
| HistGradientBoosting | 0.723 |
| LightGBM | 0.722 |
| ExtraTrees | 0.730 |
| Blend of the three | **0.715** |

`predict.py` also adds CatBoost to the blend. For reference, always predicting the average gives 0.99, and a single unregularized model on one 80/20 split gave 0.75.

The live leaderboard uses a small slice of hidden projects, so it is noisier than cross-validation. Re-sampling my own cross-validation predictions into 150-project slices gives scores anywhere between about 0.63 and 0.80.

## What I tried

- A default gradient boosting model overfit. Small trees plus strong regularization cut 5-fold CV from about 0.742 to 0.723.
- Ratio features made almost no difference on their own.
- Blending different kinds of models gave the best CV (0.715).
- TabPFN looked promising, but it needs internet or large model files at run time, and `predict.py` runs offline, so I skipped it.

## Run it

`train.csv` is not in this repo (it is Hack Club's data). Download it from the Haggle Data tab and put it next to `predict.py`, then add a `test.csv` with the same columns minus `hours_spent`, `log_hours` and `hours_bucket`:

```
python predict.py
```

This writes `predictions.csv` with one `project_id,hours_spent` row per project in `test.csv`. It needs pandas, numpy, scikit-learn, LightGBM and CatBoost, which Haggle's grader already has.

## Files

- `predict.py`: the final model and the file Haggle runs
- `main.ipynb`: my notebook for exploring the data and checking scores
- `.gitignore`: keeps the data and my virtual environment out of the repo

I used AI assistants (Claude and Google's Gemini) to help write and debug the code.
