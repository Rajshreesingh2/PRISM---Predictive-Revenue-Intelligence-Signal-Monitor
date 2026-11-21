# Fixes made after the project review

- **Train/test split:** replaced the sort-by-tenure split (train churn 31.5%, test churn 6.6%) with a stratified random split (26.5% in both). Saved the test row ids in `data/split_indices.json`.
- **Row alignment bug:** the old split reset the row index, so monthly charges, survival and archetype values were attached to the wrong customers. The original index is now kept.
- **One set of metrics:** monitoring, pipeline and API now evaluate only on the held-out test rows, with the same cut-off as phase 3. The old values (0.794, 0.858 and 0.8559) came from different evaluations, some of them on training rows.
- **Reproducibility:** `predictions_with_roi.csv` and `best_model.pkl` were not produced by any script. Phase 3 now creates both.
- **Precision and recall of 0:** each model now gets a cut-off chosen on training data (the old fixed 0.5 gave zeros on the old test set).
- **A/B test:** labelled as a simulation everywhere; the annual dollar claim was removed.
- **Bugs fixed on the way:** EDA script used column names that no longer exist and a removed matplotlib argument; SQL query broke on newer DuckDB; API fed the model the wrong feature list.
- **Phase 1 now runs.** The raw IBM Telco CSV (public, same 7,043 customers) is in `data/raw/`. Re-running phase 1 reproduces your data files exactly, apart from the 10 macro columns, which now reuse your saved real API values when offline instead of hard-coded guesses.
- **Constant features removed:** all 10 macro columns hold one value for every customer, and one of them (`trend_switch_carrier`) had reached the model's feature set. Phase 1 now drops constant columns before feature selection (AUC did not change, so they carried no information).
- **Missing libraries:** `statsmodels` and `requests` (needed by phase 1) are added to `requirements.txt`.
- **VIF filter fixed:** it used to remove every high-VIF feature in one go and only looked at the first 30 columns. It now checks all features and removes them one at a time, recomputing after each removal, until every VIF is 10 or less (30 features kept). Test AUC stayed at about 0.845.
- **Model choice:** the best model is now picked by cross-validation on the training data, not by test AUC, so the test score is no longer slightly optimistic. Logistic regression won, so it is bundled with its scaler into one saved model that works on raw inputs (without this, the API and monitoring would have given wrong numbers).
- **Reproducibility:** the Optuna search now has a fixed seed.
- **Housekeeping:** one README, a complete `requirements.txt`, `run_all.sh`, `tests/`. Removed the duplicate `src/` pipeline, `fix.py`, `generate_readme.py`, a duplicate phase 2c file, the second README, and 4 charts from an older synthetic dataset. The Excel and Tableau exporters were in `src/export/`; they are still in your git history.
