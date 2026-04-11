# SmartSort: подробная нейросетевая документация для защиты

## 1. Цель нейросетевой части

Цель проекта SmartSort в рамках дисциплины «Нейронные сети»:

- реализовать многоклассовую классификацию файлов на 8 классов (`Code`, `Docs`, `Finance`, `Study`, `Media`, `Archives`, `Logs`, `Other`);
- обучить и сравнить несколько моделей;
- встроить обученную модель в рабочее desktop-приложение;
- обеспечить воспроизводимость экспериментов и метрик.

Ключевая инженерная идея: объединить нейросетевую классификацию текста и rule-based правила по расширениям файла.

## 2. Какие модели используются

В проекте используются 3 профиля модели:

1. `legacy_mlp`  
TF-IDF + MLP (классический baseline, быстро работает на CPU).

2. `transformer_distilbert`  
`distilbert/distilbert-base-multilingual-cased` (базовый multilingual transformer).

3. `transformer_rubert`  
`DeepPavlov/rubert-base-cased` (RU-ориентированный transformer, кандидат на лучшее качество для русских текстов).

В приложении профили переключаются в `Settings -> Model profile`.

## 3. Данные и разметка

Источник датасета: объединение открытых наборов данных + локальные RU seed-корпуса + синтетические данные для некоторых классов.

Основной manifest для RU-сценария:

- `data/open_dataset_manifest.ru_extended.json`

Формат итогового датасета:

- `filename`
- `content`
- `label`

Итоговый CSV:

- `data/dataset.csv`

## 4. Формирование входа в нейросеть

Для transformer-моделей используется объединенный текст:

`[FILE] <normalized filename> [CONTENT] <content>`

Нормализация имени файла:

- lowercase;
- `_` и `-` заменяются пробелами;
- расширение включается как отдельный токен (`EXT_pdf`, `EXT_docx`, ...).

Это добавляет в модель сигнал от имени файла и расширения, а не только от текста содержимого.

## 5. Обучение моделей

### 5.1 Legacy baseline

Команда:

```powershell
python -m src.train_model
```

Артефакт:

- `models/smartsort_model.pkl`

### 5.2 DistilBERT

Команда:

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

### 5.3 ruBERT

Команда:

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

Примечание: `--no-auto-quick` отключает автоматическое упрощение режима на CPU и позволяет получить более честное качество.

## 6. Что происходит во время fine-tuning

Скрипт: `src/train_transformer_model.py`.

Pipeline:

1. Загрузка и валидация `dataset.csv`.
2. Построение `label2id` / `id2label`.
3. Stratified split train/validation.
4. Tokenization + DataLoader.
5. Fine-tuning через AdamW + scheduler warmup.
6. Валидация после каждой эпохи.
7. Early stopping по `macro_f1`.
8. Сохранение лучшей модели и отчетов.

Сохраняемые артефакты:

- модель: `models/smartsort_transformer_*/`
- метрики: `reports/model_eval/transformer_*/`
  - `summary.json`
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `history.csv`

## 7. Метрики для отчета и защиты

Основные метрики:

- Accuracy
- Macro F1 (главная метрика для несбалансированных классов)
- Weighted F1
- Confusion Matrix

Для защиты рекомендуется оформить сравнение в таблице:

| Модель | Accuracy | Macro F1 | Weighted F1 | Комментарий |
|---|---:|---:|---:|---|
| Legacy TF-IDF + MLP | ... | ... | ... | быстрый baseline |
| DistilBERT | ... | ... | ... | компромисс скорость/качество |
| ruBERT | ... | ... | ... | основной quality-кандидат для RU |

## 8. Runtime-архитектура (почему это «прикладная НС-система»)

В runtime используются 2 уровня:

1. Rule-based приоритет по расширениям файла (`Media`, `Archives`, `Code`, `Logs`).
2. Нейросетевая классификация для остальных случаев (`Docs`, `Finance`, `Study`, часть `Other`).

Дополнительно применяется confidence threshold (по умолчанию `0.55`):

- если уверенность ниже порога -> класс `Other`.

Это повышает устойчивость в реальных условиях и снижает риск ошибочного авто-перемещения.

## 9. Что показать на защите пошагово

1. Открыть `Settings` и переключить профиль модели (legacy/distilbert/rubert).
2. Показать, что профиль сохраняется в `config.json` (`selected_model_key`).
3. Открыть `reports/model_eval/transformer_distilbert/summary.json` и `reports/model_eval/transformer_rubert/summary.json`.
4. Показать `confusion_matrix.csv` и объяснить типичные ошибки.
5. Запустить live-demo сортировки в UI.
6. Показать, что есть fallback/rule-based устойчивость и undo.

## 10. Частые вопросы комиссии (и короткие ответы)

Почему не только классический ML?

- Transformer лучше учитывает контекст и сложные формулировки в RU/EN документах.

Зачем тогда legacy-модель?

- Это baseline для сравнения и быстрая резервная модель.

Почему ruBERT может быть лучше DistilBERT?

- ruBERT дообучался под русский язык и обычно лучше в русскоязычных формулировках.

Почему hybrid (rules + NN)?

- Чисто нейросетевая схема хуже на бинарных/служебных файлах; rules повышают надежность.

## 11. Мини-чеклист перед финальной сдачей

1. Дообучить все 3 профиля и зафиксировать отчеты.
2. Подготовить итоговую таблицу сравнения моделей.
3. Сохранить скриншоты/логи демонстрации.
4. Проверить, что в UI корректно переключается профиль и сохраняются настройки.
5. Добавить ссылки на итоговые отчеты в презентацию/пояснительную записку.
