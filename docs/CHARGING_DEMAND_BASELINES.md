# Regional charging demand sequence baselines

This is a separate forecasting branch of the charging-data research. It does not reproduce AgentMove, use LLMs, or establish a memory/RL benefit. Customer event data, region-level time series, checkpoints and individual predictions remain external to Git.

## Input contract

Use the locally accepted `grid_2km_v1` dataset, including `PROTOCOL.json`, `VERIFIED.json`, `series.npz`, `events.npz`, `windows.csv`, `normalization_train_only.npz`, and `baselines_validation/`. The pre-existing dataset builder is not bundled or implicitly republished here. The new reader accepts the validated NPZ/CSV schema, records hashes, and verifies causal baseline parity against the dataset's original controls.

The two targets are hourly valid order starts and the final session energy of those orders attributed to their start hour. The latter is **not hourly metered load**. No VINs or user-level records are needed. Historical cohorts include only orders completed by the forecast origin. Do not substitute finalized recent-hour labels for historical inputs.

The core contains 112 training-selected approximately 2 km grid regions. Lookback 168 hours, horizon 24 hours, stride one hour. Train 2025-11-01 through 12-15, validation 12-16 through 12-31, test 2026-01-01 through 01-13. The 24-hour label-maturation buffer also fits inside each split. Origin counts are 865/337/265. Test data were previously explored in aggregate under another individual-station protocol; this is not a pristine blind benchmark.

## Models and selection

- Four controls: zero, previous day, previous week, training hour-of-week mean.
- Per-region Ridge autoregression: all 168 hourly lags of both channels plus calendar dummies, direct 24-step outputs. Alpha selected from 1/10/100/1000 by validation.
- Joint-region MLP: flattened historical sequence, width128.
- Joint-region GRU and LSTM: single layer, width64, direct multistep decoder.
- Neural models include known past/future calendar and history-maturation indicators; use train-only raw-value scales and nonnegative Softplus outputs. MSE training balances the two channels by their global training standard deviations. This experiment-specific raw scaler does not overwrite the dataset's log scaler.
- Three seeds 17/29/43, at most30 epochs, patience5, AdamW learning rate0.001, weight decay0.0001, dropout0.1, batch32. Epoch/alpha selection averages the two validation MAEs relative to the fixed hour-of-week baseline. Family seed means and prediction ensembles are reported separately. The arithmetic three-seed ensemble is predeclared, not test-fitted.

The compact implementations are baseline experiments, not exact reproductions of particular papers. API references: [PyTorch2.8 GRU](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html), [PyTorch2.8 LSTM](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html), [Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). Ridge is implemented with NumPy normal equations and regularization; scikit-learn is not required.

## Reproduce

Verified runtime: Python3.9.7, NumPy1.22.4, Pandas1.3.4, PyTorch2.8.0+cpu; Matplotlib for figures. No network/model-service calls. Set `CHARGING_DATA` to the external accepted dataset and `CHARGING_RUN` to a new external output directory. From repository root, using the Python environment containing these packages:

```powershell
python scripts/charging_demand/run.py --stage develop --data $env:CHARGING_DATA --config configs/charging_demand_baselines_v1.json --output $env:CHARGING_RUN
python scripts/charging_demand/verify.py --split validation --data $env:CHARGING_DATA --output $env:CHARGING_RUN
python scripts/charging_demand/run.py --stage test --data $env:CHARGING_DATA --config configs/charging_demand_baselines_v1.json --output $env:CHARGING_RUN
python scripts/charging_demand/verify.py --split test --data $env:CHARGING_DATA --output $env:CHARGING_RUN
python scripts/report_charging_demand.py --run $env:CHARGING_RUN
```

Development refuses to overwrite an output directory. Test requires an independently checked development stage plus unchanged code, data, configuration and checkpoint hashes. All selected configurations are sealed in `FINAL_SELECTION.json` **before** test window construction; no validation refitting or post-test tuning. Do not alter source files between development and test, or reuse a scored test directory for development. A running lock and per-epoch logs identify the one CPU training process; timeout is3hours per stage and BLAS/PyTorch threads are limited to1. Failure receipts remain on disk.

Verification recomputes metrics independently, re-infers three origins per checkpoint, checks the saved ensemble, checks epoch selection, compares two event aggregation paths and tests future-completion isolation. This does not rerun all training or guarantee cross-environment determinism. Metrics: MAE/RMSE/WAPE, positive-target MAE, bias, region MAE, and leads1/3/6/12/24. No zero-dividing MAPE. Moving-block bootstrap intervals (48h blocks,2000 draws) are exploratory due to the short test period.

Source is committed here; generated run outputs remain outside this repository. Only small aggregate result summaries may be published after verification.

## Accepted 2026-10-22 run

See [the aggregate results and verification receipts](../reports/charging_demand_sequence_20261022.json) and [the maintained work log](WORK_LOG.md#6-区域充电需求预测独立于agentmove的任务). The validation-selected three-seed MLP prediction ensemble achieved test MAE0.6639 orders and29.3442kWh, reductions of2.04% and3.54% versus the hour-of-week mean. Its RMSE and positive-demand-hour MAE were worse than that baseline; Ridge had the lowest RMSE among these configurations. Several recurrent checkpoints reached the30-epoch cap, so this comparison does not establish performance after full convergence/tuning. All test outcomes remain frozen.

`scripts/finalize_charging_demand_delivery.py` preserves the one-off, standard-library delivery procedure for this exact sealed run. It appends the accepted evidence to the work log and publishes only an aggregate summary. It is hash-bound to that run and refuses to overwrite its public summary; it is not part of the general model-training procedure. The original invocation used Python3.12 after Python3.9 rejected a long Chinese source line during parsing; no training or scoring failed or was repeated for this reporting issue. For new runs, use the generic reporter above and review the aggregate claims separately.
