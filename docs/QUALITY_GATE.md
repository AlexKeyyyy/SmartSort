# SmartSort Quality Gate

This checklist is the minimal gate before final submission and demo.

## 1) ML quality artifacts

- Run `python -m src.evaluate_model`
- Confirm files exist in `reports/model_eval/latest/`:
  - `summary.json`
  - `classification_report.txt`
  - `classification_report.json`
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `top_errors.csv`

## 2) Smoke checks

- Run `python -m src.smoke_checks`
- Expected result: `status = ok` in `reports/smoke/latest/smoke_report.json`
- Optional extended check:
  - `python -m src.smoke_checks --run-training-check`

## 3) Runtime stability

- App starts with `python main.py`
- Watcher `Start/Stop` works and status indicator updates correctly
- `Apply all` moves pending files
- `Undo` restores the latest moved file and does not crash on repeated call
- Settings are persisted to `config.json`

## 4) Demo readiness

- Prepare demo input files:
  - `python -m src.prepare_demo_pack`
- Verify `data/demo_pack/input` contains files for all 8 categories
- Verify `data/demo_pack/expected_labels.json` is ready for live comparison

## 5) Final acceptance criteria

- No critical runtime crashes during 7-10 minute live demo
- 8 categories are shown in dashboard/watcher flow
- Model quality artifacts are attached in report/presentation
- Team can explain architecture, metrics, and error analysis
