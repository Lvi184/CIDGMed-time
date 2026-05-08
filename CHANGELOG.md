# Changelog

## Enhanced version

- Added stage duration target `Y_step_duration.npy`.
- Extended `StepCIDGMedNet` with `duration_head`.
- Updated training to jointly predict medication regimen, LOS, and step duration.
- Updated `prediction_preview.csv` export to include:
  - `predicted_los`
  - `true_los`
  - `predicted_step_duration`
  - `true_step_duration`
  - `predicted_med_regimen`
  - `predicted_med_probabilities`
  - `true_med_regimen`
- Added `src/recommend_regimen.py` to combine model medication probabilities with causal ATE estimates.
- Added `scripts/run_pipeline.sh` for one-command execution.
- Updated README with the new workflow and output definitions.

## Evaluation Metrics Upgrade

- Added `src/evaluate_project.py` for end-to-end evaluation.
- Added multi-label medication metrics: Micro/Macro F1, Jaccard, Hamming Loss, Exact Match, Recall@K, Precision@K, HitRate@K, AUROC, AUPRC, LRAP.
- Added regression metrics for LOS and step duration: MAE, Median AE, RMSE, R², MAPE, bias.
- Added causal diagnostics: SMD before/after IPW, effective sample size, propensity overlap summary, bootstrap 95% CI for ATE.
- Added per-medication metric table.
- Added evaluation plots for prediction quality, medication probability separation, and causal balance.
- Updated `scripts/run_pipeline.sh` to run evaluation automatically after training and recommendation.
- Updated training script to save train/validation indices for clean validation-set evaluation.
