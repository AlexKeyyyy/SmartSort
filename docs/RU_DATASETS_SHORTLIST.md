# RU Datasets Shortlist For SmartSort

Цель: усилить русскоязычную часть данных, особенно для классов `Docs`, `Finance`, `Study`, без ухудшения общего качества.

## Таблица кандидатов

| Dataset | Link | Target class in SmartSort | Candidate fields | Risks/notes |
|---|---|---|---|---|
| IlyaGusev/ru_news | https://huggingface.co/datasets/IlyaGusev/ru_news | Docs | `title`, `summary`, `text` | В `datasets>=4` требует legacy script (`ru_news.py`), используйте только при downgrade |
| IlyaGusev/habr | https://huggingface.co/datasets/IlyaGusev/habr | Docs / Study | `title`, `text`, `tags` | Технические статьи могут частично пересекаться с Code |
| IlyaGusev/ru_stackoverflow | https://huggingface.co/datasets/IlyaGusev/ru_stackoverflow | Code / Study | `title`, `question`, `answers`, `tags` | В `datasets>=4` требует legacy script (`ru_stackoverflow.py`), используйте только при downgrade |
| deepvk/cultura_ru_edu | https://huggingface.co/datasets/deepvk/cultura_ru_edu | Study / Docs | `title`, `text`, `content`, `abstract` | Очень большой набор, начинать с небольшого sample_limit |
| mlsa-iai-msu-lab/ru_sci_bench_mteb | https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb | Study | `title`, `abstract`, `text` | На странице указано ~182k научных title+abstract, полезно для Study |
| slm-ct/fas_ad_practice_dataset | https://huggingface.co/datasets/slm-ct/fas_ad_practice_dataset | Docs / Finance | - | В текущем формате датасет отдает метаданные, а не текстовый корпус; в training manifest исключен |
| Kasymkhan/RussianFinancialNews | https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews | Finance | `title`, `body`, `tags` | Уже используется в базовом manifest, оставить как базовый finance-anchor |
| IlyaGusev/gazeta | https://huggingface.co/datasets/IlyaGusev/gazeta | Docs | `title`, `summary`, `text` | Уже используется в базовом manifest |

## Протокол отбора (обязательный)

Для каждого кандидата:

1. Проверить, что из полей реально собирается текст (`build_open_dataset` не должен собирать пустые строки).
2. Поставить разумный `sample_limit` для trial-run.
3. Запустить pipeline и получить артефакты.
4. Сравнить `classification_report` и `top_errors` с baseline.
5. Оставить датасет, если он улучшает/не ухудшает классы `Docs/Finance/Study`.

## Решение по лицензиям

На этом этапе shortlist собирается по релевантности.
Перед финальной сдачей сделать отдельную лицензионную проверку и зафиксировать ее в отчете.
