# SmartSort: Полный Разбор С Нуля

Этот документ для ситуации "я вообще ноль". Цель: понять, что за проект, как он работает как приложение, где здесь нейросеть и почему это не просто "скрипт с моделью".

## 1) Что это за проект простыми словами

`SmartSort` — десктопное приложение, которое автоматически сортирует файлы по папкам.

Пользовательский сценарий:

1. Вы выбираете папку, за которой нужно следить (`watch_dir`).
2. Вы выбираете папку, куда складывать отсортированные файлы (`output_dir`).
3. Приложение видит новый файл.
4. Модель определяет категорию (`Code`, `Docs`, `Finance`, `Study`, `Media`, `Archives`, `Logs`, `Other`) и уверенность (`confidence`).
5. Файл можно переместить автоматически или вручную.
6. Действие пишется в базу, и его можно отменить через `Undo`.

## 2) Почему это полноценное приложение, а не "только нейросеть"

В проекте есть 3 слоя:

- `ML layer`: данные, обучение, оценка модели.
- `Core layer`: watcher, predictor, mover, storage.
- `UI layer`: экран, кнопки, статусы, лог, настройки.

Нейросеть — это только часть `predictor`.
Остальное — инфраструктура, runtime, UX и надежность.

## 3) Из каких модулей состоит система и зачем каждый

### UI

- `src/app.py`
  - вкладки: `Dashboard`, `Watcher`, `Log`, `Settings`
  - команды пользователя: `Start/Stop`, `Apply all`, `Undo`, `Save settings`
  - показывает категории, confidence, историю операций

### Core

- `src/watcher.py`
  - следит за папкой и ловит новые файлы (watchdog)
- `src/predictor.py`
  - получает файл и предсказывает категорию
  - для `Media` и `Archives` использует extension fallback (правила)
- `src/mover.py`
  - физически перемещает файл в целевую папку
  - поддерживает `undo_last`
- `src/storage.py`
  - хранит журнал перемещений в `SQLite` (`smartsort.db`)
  - отдает статистику и лог
- `src/extractor.py`
  - извлекает текст из `txt/pdf/docx/xlsx/...`

### ML

- `src/generate_dataset.py`
  - синтетический датасет
- `src/build_open_dataset.py`
  - сбор датасета из open datasets по manifest JSON
- `src/train_model.py`
  - обучение `TF-IDF + MLPClassifier`
- `src/evaluate_model.py`
  - метрики и артефакты (report, confusion matrix, top errors)
- `src/train_open_model.py`
  - единый CLI пайплайн: build -> train -> evaluate -> smoke

## 4) Полный runtime поток "от файла до результата"

1. Watcher ловит новый файл.
2. Predictor строит признаки:
   - из имени файла,
   - из текста файла (если это текстовый формат).
3. Модель выдает вероятности классов и лучший класс.
4. Если confidence ниже порога, возможно fallback в `Other`.
5. Mover переносит файл в `output_dir/<Category>/`.
6. Storage пишет запись в `moves`.
7. UI обновляет таблицу и Dashboard.
8. Пользователь может нажать `Undo`.

## 5) Где тут нейросеть и какие признаки

Модель: `MLPClassifier` (многослойный персептрон).

Признаки:

- `filename_tfidf`: char n-grams (2..4) по имени файла.
- `content_word_tfidf`: word n-grams (1..2) по содержимому.
- `content_char_tfidf`: char n-grams (3..5) по содержимому.

Плюс нормализация имени (например, добавление токена расширения `EXT_pdf`).

## 6) Как читать метрики и доказать качество

Ключевые артефакты лежат в `reports/model_eval/latest/`:

- `summary.json` — главное резюме.
- `classification_report.txt/csv/json` — precision/recall/f1.
- `confusion_matrix.csv` — кто с кем путается.
- `top_errors.csv` — примеры ошибок для анализа.

Для защиты важно:

- показать общую accuracy;
- показать, что вы анализируете не только accuracy, но и ошибки по классам.

## 7) Надежность и проверяемость (не ML, но критично для оценки)

`python -m src.smoke_checks` проверяет:

- predictor (корректные выходы),
- watcher (start/stop, событие),
- mover/storage/undo,
- сохранение и чтение `config.json`.

Это аргумент "система стабильная", а не "у нас просто один хороший запуск".

## 8) Как запускать end-to-end

### Базовый запуск приложения

```bash
python -m pip install -r requirements.txt
python main.py
```

### Полный pipeline обучения и проверки

```bash
python -m src.train_open_model \
  --manifest data/open_dataset_manifest.example.json \
  --evaluate \
  --smoke-checks
```

## 9) Что говорить преподавателю в одном абзаце

"SmartSort — это не только модель классификации. Это полноценное desktop-приложение с архитектурой из ML/Core/UI слоев: watcher отслеживает новые файлы, predictor классифицирует их по имени и содержимому, mover перемещает, storage логирует в SQLite и поддерживает undo, а UI дает управляемый пользовательский сценарий. Качество подтверждается reproducible артефактами оценки и smoke-тестами."
