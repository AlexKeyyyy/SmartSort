from __future__ import annotations

import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RANDOM_SEED = 42
SAMPLES_PER_LABEL = 180


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_docs_rows() -> list[dict[str, str]]:
    filenames = [
        "служебная_записка",
        "новости_отдела",
        "план_проекта",
        "протокол_встречи",
        "заметки_команды",
        "описание_процесса",
    ]
    subjects = [
        "запуск нового сервиса",
        "обновление внутреннего регламента",
        "подготовка презентации для партнеров",
        "обсуждение сроков релиза",
        "согласование требований по проекту",
        "изменение графика тестирования",
    ]
    actions = [
        "команда согласовала список задач и ответственных",
        "в документ внесли замечания по качеству и срокам",
        "руководитель попросил обновить описание этапов работ",
        "участники встречи зафиксировали риски и дальнейшие шаги",
        "в тексте указаны основные выводы и рекомендации",
    ]
    endings = [
        "Материал подготовлен для внутреннего использования и обсуждения.",
        "Документ нужен для координации команды и планирования следующих задач.",
        "Файл содержит общую информацию без финансовой отчетности и формул.",
    ]

    rows: list[dict[str, str]] = []
    for index in range(SAMPLES_PER_LABEL):
        subject = subjects[index % len(subjects)]
        filename = f"{filenames[index % len(filenames)]}_{index}.txt"
        content = (
            f"Текстовый документ по теме: {subject}. "
            f"{actions[index % len(actions)]}. "
            f"{endings[index % len(endings)]}"
        )
        rows.append({"filename": filename, "content": content})
    return rows


def build_finance_rows() -> list[dict[str, str]]:
    documents = [
        "счет",
        "акт",
        "накладная",
        "платежное_поручение",
        "отчет_по_бюджету",
        "ведомость",
    ]
    departments = [
        "отдел закупок",
        "финансовый департамент",
        "бухгалтерия",
        "коммерческий отдел",
    ]
    vendors = [
        "ООО Альфа",
        "ООО Вектор",
        "АО Спектр",
        "ИП Соколова",
    ]
    purposes = [
        "оплата поставки оборудования",
        "закрытие услуг по сопровождению",
        "перечисление аванса по договору",
        "подтверждение выполнения работ",
    ]

    rows: list[dict[str, str]] = []
    for index in range(SAMPLES_PER_LABEL):
        amount = 15000 + index * 137
        vat = round(amount * 0.2, 2)
        filename = f"{documents[index % len(documents)]}_{index}.txt"
        content = (
            f"{documents[index % len(documents)].replace('_', ' ').capitalize()} №{1000 + index} от 08.04.2026. "
            f"Контрагент: {vendors[index % len(vendors)]}. "
            f"Подразделение: {departments[index % len(departments)]}. "
            f"Назначение платежа: {purposes[index % len(purposes)]}. "
            f"Сумма к оплате {amount} руб., НДС {vat} руб., срок оплаты 5 банковских дней."
        )
        rows.append({"filename": filename, "content": content})
    return rows


def build_study_rows() -> list[dict[str, str]]:
    filenames = [
        "лекция_нейросети",
        "конспект_линейная_алгебра",
        "лабораторная_физика",
        "методичка_матанализ",
        "семинар_машинное_обучение",
        "курсовая_черновик",
    ]
    topics = [
        "градиентный спуск и функция потерь",
        "собственные значения и векторы",
        "метод наименьших квадратов",
        "переобучение и регуляризация",
        "обратное распространение ошибки",
        "классификация текстов и TF-IDF признаки",
    ]
    tasks = [
        "В тексте объясняются основные определения, формулы и примеры решения задач.",
        "Материал подготовлен для экзамена и содержит теоретические выводы.",
        "Документ описывает лабораторную работу, цель исследования и порядок расчета.",
        "В заметках есть ссылки на лекцию, домашнее задание и рекомендуемую литературу.",
    ]

    rows: list[dict[str, str]] = []
    for index in range(SAMPLES_PER_LABEL):
        filename = f"{filenames[index % len(filenames)]}_{index}.txt"
        content = (
            f"Учебный материал по теме: {topics[index % len(topics)]}. "
            f"{tasks[index % len(tasks)]} "
            f"Используются понятия матрицы, производная, эксперимент, модель и вычисления."
        )
        rows.append({"filename": filename, "content": content})
    return rows


def main() -> None:
    random.seed(RANDOM_SEED)

    datasets = {
        RAW_DIR / "russian_docs.jsonl": build_docs_rows(),
        RAW_DIR / "russian_finance.jsonl": build_finance_rows(),
        RAW_DIR / "russian_study.jsonl": build_study_rows(),
    }

    for path, rows in datasets.items():
        random.shuffle(rows)
        write_jsonl(path, rows)
        print(f"[OK] Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
