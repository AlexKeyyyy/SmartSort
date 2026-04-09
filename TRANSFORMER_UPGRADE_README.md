# SmartSort transformer upgrade

This folder adds a stronger NLP baseline/final model for SmartSort using a pretrained multilingual transformer.

## Files
- `train_transformer_model.py` — fine-tunes a Hugging Face sequence-classification model on `dataset.csv`
- `predict_transformer.py` — predicts one file class from filename + content
- `train_open_transformer_model.py` — end-to-end orchestration: raw RU corpus -> merged dataset -> transformer fine-tuning

## Recommended first run

```bash
pip install -r requirements.txt
python -m src.train_open_transformer_model \
  --model-name distilbert/distilbert-base-multilingual-cased \
  --quick
```

`--quick` enables a speed-first preset (shorter sequence, fewer epochs, subset sampling, frozen backbone).
On CPU this drastically reduces training time while keeping decent baseline quality.

## Better Russian-focused alternative
If you want a stronger Russian model and have enough VRAM/RAM, try:

```bash
python -m src.train_open_transformer_model \
  --model-name DeepPavlov/rubert-base-cased \
  --epochs 3 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --max-length 256
```

## Progress visibility

`src.train_transformer_model` now shows per-epoch tqdm progress bars and periodic live metrics:
- running train loss
- current learning rate
- epoch validation metrics and epoch time

Useful knobs:
- `--log-interval 10`
- `--max-train-samples 3000 --max-val-samples 1000`
- `--no-auto-quick` to disable automatic CPU quick mode

## Inference example

```bash
python -m src.predict_transformer \
  --filename "invoice_april_15.pdf" \
  --content "Сумма к оплате 15200 руб. НДС 20 процентов. Оплата в течение 5 банковских дней."
```

## Outputs
- model weights/tokenizer: `models/smartsort_transformer/`
- eval artifacts: `reports/model_eval/transformer/`
  - `summary.json`
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `history.csv`
