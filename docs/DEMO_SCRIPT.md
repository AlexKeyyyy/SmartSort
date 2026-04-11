# SmartSort Demo Script (7-10 min)

## 0:00-1:00 Intro

- Goal: intelligent desktop file sorting with ML + rules
- Categories: `Code`, `Docs`, `Finance`, `Study`, `Media`, `Archives`, `Logs`, `Other`
- Stack: Python, scikit-learn, CustomTkinter, watchdog, SQLite

## 1:00-2:30 Architecture snapshot

- Show `ARCHITECTURE.md`
- Mention pipeline:
  - watcher detects file
  - predictor classifies with confidence
  - mover sends to category folder
  - storage logs operation and supports undo

## 2:30-4:00 Model quality

- Open `reports/model_eval/latest/summary.json`
- Show `classification_report.txt` and `confusion_matrix.csv`
- Briefly explain `top_errors.csv` and what kinds of mistakes remain

## 4:00-7:00 Live app run

- Launch app: `python main.py`
- Set watch/output folders
- Start watcher
- Add files from `data/demo_pack/input` into watch folder
- Show predictions and confidence in watcher table
- Click `Apply all` (or use auto-move)
- Show dashboard and log updates
- Run `Undo` once and show restored file

## 7:00-8:30 Reliability and checks

- Show `reports/smoke/latest/smoke_report.json`
- Explain smoke coverage: predictor, watcher, mover/storage, config, optional training

## 8:30-10:00 Wrap-up

- Constraints and known limitations
- Improvement roadmap
- Team ownership split and who implemented what
