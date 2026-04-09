# SmartSort

SmartSort is a desktop application for intelligent file sorting.

## What it does

- watches a selected folder in real time
- extracts filename and file content features
- classifies files into 8 categories with an ML model
- moves files into category folders
- stores move history in SQLite
- supports undo for the latest move

## Categories

- Code
- Docs
- Finance
- Study
- Media
- Archives
- Logs
- Other

## Project structure

```text
data/           generated dataset
models/         trained ML model
src/            application source code
config.json     app settings
main.py         entry point
requirements.txt
```

## Main modules

- `src/generate_dataset.py` - synthetic dataset generation
- `src/build_open_dataset.py` - converts open datasets to the SmartSort CSV format
- `src/train_open_model.py` - full pipeline: build dataset from open sources and train the model
- `src/train_model.py` - model training and evaluation
- `src/evaluate_model.py` - reproducible model evaluation artifacts (accuracy/report/confusion/errors)
- `src/smoke_checks.py` - smoke checks for predictor/watcher/mover/storage/config
- `src/prepare_demo_pack.py` - creates a clean demo file set for live presentation
- `src/compare_evaluations.py` - compares baseline vs candidate evaluation artifacts
- `src/extractor.py` - text extraction from supported file types
- `src/predictor.py` - file classification
- `src/storage.py` - SQLite move log and stats
- `src/mover.py` - moving files and undo
- `src/watcher.py` - folder monitoring via watchdog
- `src/app.py` - CustomTkinter desktop UI

## Official training pipeline (submission mode)

SmartSort expects a CSV with three columns:

- `filename`
- `content`
- `label`

You can build this CSV from open datasets listed in `data/open_dataset_manifest.example.json`,
train, evaluate, and run smoke checks with one command:

```powershell
python -m pip install -r requirements.txt
python -m src.train_open_model ^
  --manifest data/open_dataset_manifest.example.json ^
  --evaluate ^
  --smoke-checks
```

Optional long smoke run (includes full retraining inside smoke checks):

```powershell
python -m src.train_open_model ^
  --manifest data/open_dataset_manifest.example.json ^
  --evaluate ^
  --smoke-checks ^
  --smoke-training-check
```

RU-extended experiment manifest (for stronger Russian coverage):

```powershell
set HF_HUB_DOWNLOAD_TIMEOUT=120
python -m src.train_open_model
  --manifest data/open_dataset_manifest.ru_extended.json
  --source-retries 1
  --evaluate
  --eval-output-dir reports/model_eval/ru_extended
  --smoke-checks
```

Baseline run for comparison:

```powershell
python -m src.train_open_model ^
  --manifest data/open_dataset_manifest.example.json ^
  --evaluate ^
  --eval-output-dir reports/model_eval/baseline ^
  --smoke-checks
```

Compare baseline vs RU-extended:

```powershell
python -m src.compare_evaluations ^
  --baseline-dir reports/model_eval/baseline ^
  --candidate-dir reports/model_eval/ru_extended ^
  --output reports/model_eval/comparison_ru_extended.md
```

Recommended mapping for the current categories:

- `Code` -> CodeSearchNet
- `Docs` -> AG News + Gazeta + local Russian office-style corpus
- `Finance` -> financial news datasets (EN+RU) + RussianFinancialNews + local Russian finance corpus
- `Study` -> arXiv scientific papers + ClassNotes + HumArticles + local Russian study corpus
- `Logs` -> HDFS logs
- `Media` -> keep rule-based or synthetic by file extension
- `Archives` -> keep rule-based or synthetic by file extension
- `Other` -> synthetic or mixed unknown files

The example manifest already targets a mixed RU+EN dataset and aims for:

- `Code`: 1000 rows
- `Docs`: 1000 rows
- `Finance`: 800 rows
- `Study`: 900 rows
- `Logs`: 1000 rows
- `Media`: 600 synthetic rows
- `Archives`: 600 synthetic rows
- `Other`: 600 synthetic rows

During `src.train_open_model`, the project generates a small local Russian seed corpus and combines
it with English and Russian open datasets into a single multilingual `data/dataset.csv`.

## Model profiles and switching (UI)

The app now supports profile switching in **Settings -> Model profile**.

Supported profiles:

- `legacy_mlp` -> `models/smartsort_model.pkl` (TF-IDF + MLP baseline)
- `transformer_distilbert` -> `models/smartsort_transformer_distilbert/`
- `transformer_rubert` -> `models/smartsort_transformer_rubert/`

The selected profile is saved in `config.json` as `selected_model_key`.
`Media`, `Archives`, `Code`, and `Logs` still keep rule-based extension fallbacks for stable runtime behavior.

### Train all 3 profiles (recommended for defense)

1. Legacy baseline:

```powershell
python -m src.train_model
```

2. DistilBERT multilingual transformer:

```powershell
python -m src.train_open_transformer_model ^
  --manifest data/open_dataset_manifest.ru_extended.json ^
  --model-name distilbert/distilbert-base-multilingual-cased ^
  --output-dir models/smartsort_transformer_distilbert ^
  --report-dir reports/model_eval/transformer_distilbert ^
  --source-retries 1 ^
  --epochs 3 ^
  --batch-size 8 ^
  --eval-batch-size 16 ^
  --max-length 256
```

3. ruBERT transformer:

```powershell
python -m src.train_open_transformer_model ^
  --manifest data/open_dataset_manifest.ru_extended.json ^
  --model-name DeepPavlov/rubert-base-cased ^
  --output-dir models/smartsort_transformer_rubert ^
  --report-dir reports/model_eval/transformer_rubert ^
  --source-retries 1 ^
  --epochs 3 ^
  --batch-size 4 ^
  --eval-batch-size 8 ^
  --max-length 256 ^
  --no-auto-quick
```

## Reproducible quality artifacts

Model evaluation:

```powershell
python -m src.evaluate_model
```

Output directory: `reports/model_eval/latest/`

- `summary.json`
- `classification_report.txt`
- `classification_report.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `top_errors.csv`

Smoke checks:

```powershell
python -m src.smoke_checks
```

Output file: `reports/smoke/latest/smoke_report.json`

## Demo preparation

Generate a clean demo input pack (all 8 categories):

```powershell
python -m src.prepare_demo_pack
```

Files will be created in `data/demo_pack/input/` with expected labels in
`data/demo_pack/expected_labels.json`.

## Run app (official entrypoint)

```powershell
python -m pip install -r requirements.txt
python main.py
```

## Submission checklist docs

- `docs/QUALITY_GATE.md`
- `docs/DEMO_SCRIPT.md`
- `docs/DEFENSE_ROLES.md`
- `docs/PROJECT_FROM_ZERO.md`
- `docs/DEFENSE_CHEATSHEET.md`
- `docs/RU_DATASETS_SHORTLIST.md`
- `docs/RU_DATASET_BENCHMARK_PROTOCOL.md`
- `docs/NEURAL_NETWORKS_MAX_GRADE_RU.md`
- `docs/REPORT_TEMPLATE.md`
- `docs/TRELLO_SYNC_RULES.md`
