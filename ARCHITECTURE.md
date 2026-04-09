# SmartSort Architecture

## Layers

### ML layer

- `generate_dataset.py` creates a synthetic dataset with `filename`, `content`, and `label`
- `build_open_dataset.py` converts downloaded open datasets into the same SmartSort CSV format
- `train_model.py` trains a baseline sklearn pipeline
- `predictor.py` loads the trained model and returns category and confidence

### Core layer

- `extractor.py` reads text from supported document and code formats
- `watcher.py` listens for new files in the watch folder
- `mover.py` moves files into category folders and supports undo
- `storage.py` stores move history and statistics in SQLite

### UI layer

- `app.py` provides a CustomTkinter desktop interface
- UI receives watcher events, displays predictions, launches moves in background threads, and shows logs/stats

## Data flow

1. User selects watch and output folders in the UI.
2. `FolderWatcher` detects a new file.
3. `SmartSortPredictor` extracts text and predicts category/confidence.
4. UI shows the prediction.
5. If auto-move is enabled, `FileMover` moves the file.
6. `Storage` logs the action and updates statistics.
7. User can undo the last move from Dashboard or Log.

## Storage

- `data/dataset.csv` - training dataset
- `models/smartsort_model.pkl` - trained model
- `smartsort.db` - SQLite move history
- `config.json` - application settings
