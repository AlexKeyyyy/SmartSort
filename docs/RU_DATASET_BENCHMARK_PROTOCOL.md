# RU Dataset Benchmark Protocol (Baseline vs Extended)

Этот протокол нужен, чтобы решение по новым датасетам было инженерным, а не "на глаз".

## 1) Что сравниваем

- `baseline`: текущий `data/open_dataset_manifest.example.json`
- `extended`: новый `data/open_dataset_manifest.ru_extended.json`

## 2) Как запускать

### Baseline

```bash
python -m src.train_open_model \
  --manifest data/open_dataset_manifest.example.json \
  --evaluate \
  --eval-output-dir reports/model_eval/baseline \
  --smoke-checks
```

### RU Extended

```bash
python -m src.train_open_model \
  --manifest data/open_dataset_manifest.ru_extended.json \
  --evaluate \
  --eval-output-dir reports/model_eval/ru_extended \
  --smoke-checks
```

## 3) Что фиксировать после каждого прогона

- `summary.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `top_errors.csv`
- `smoke_report.json` (должен быть `status = ok`)

## 4) Критерии принятия RU-расширения

RU-расширение принимается, если одновременно:

1. Не падает smoke-check pipeline.
2. Нет значимого ухудшения по `macro avg f1` и `weighted avg f1`.
3. Для целевых классов (`Docs`, `Finance`, `Study`) качество улучшается или остается стабильным.
4. `top_errors.csv` не показывает систематический новый тип ошибок.

## 5) Быстрый шаблон вывода в отчет

- "Добавление RU datasets X/Y/Z дало прирост по классу Study на N пунктов recall."
- "По Finance precision осталась стабильной, но снизился recall на M пунктов, поэтому датасет Z исключен."
- "Итоговый extended manifest включает только источники, прошедшие benchmark."

## 6) Командная ответственность

- Артём: сравнение метрик и ошибок.
- Алексей: фиксация решений и артефактов в отчете/Trello.
- Светлана: проверка, что runtime/UX не деградировали после новых моделей.
