from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
SAMPLES_PER_CLASS = 600
RANDOM_SEED = 42


def build_code_sample(index: int) -> tuple[str, str]:
    options = [
        (
            f"api_handler_{index}.py",
            "def fetch_user(user_id):\n"
            "    query = 'SELECT id, email FROM users WHERE id = %s'\n"
            "    return db.execute(query, (user_id,))\n"
            "class UserService:\n"
            "    def validate_token(self, token):\n"
            "        return token.startswith('sk_')\n",
        ),
        (
            f"frontend_widget_{index}.js",
            "export function renderWidget(items) {\n"
            "  return items.map((item) => `<li>${item.name}</li>`).join('');\n"
            "}\n"
            "const state = { loading: false, page: 1 };\n",
        ),
        (
            f"migration_{index}.sql",
            "CREATE TABLE invoices (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    customer_name TEXT,\n"
            "    total_amount DECIMAL(10, 2)\n"
            ");\n"
            "CREATE INDEX idx_invoices_customer ON invoices(customer_name);\n",
        ),
        (
            f"engine_core_{index}.cpp",
            "#include <vector>\n"
            "#include <string>\n"
            "int compute_checksum(const std::vector<int>& values) {\n"
            "    int result = 0;\n"
            "    for (int value : values) { result ^= value; }\n"
            "    return result;\n"
            "}\n",
        ),
    ]
    return random.choice(options)


def build_docs_sample(index: int) -> tuple[str, str]:
    options = [
        (
            f"meeting_notes_{index}.txt",
            "Meeting notes: reviewed release scope, assigned QA owners, "
            "documented deployment risks, and scheduled follow-up with platform team.",
        ),
        (
            f"technical_spec_{index}.md",
            "Technical specification for the ingestion service. The module exposes REST endpoints, "
            "stores audit events, and retries failed deliveries with exponential backoff.",
        ),
        (
            f"user_manual_{index}.docx",
            "User manual draft describing login flow, navigation sidebar, search filters, "
            "troubleshooting steps, and administrator permissions.",
        ),
        (
            f"architecture_overview_{index}.pdf",
            "System overview with service boundaries, API contracts, integration sequence, "
            "dependency notes, and acceptance criteria for the current milestone.",
        ),
    ]
    return random.choice(options)


def build_finance_sample(index: int) -> tuple[str, str]:
    month = random.choice(["jan", "feb", "mar", "apr", "may", "jun"])
    amount = random.randint(1200, 25000)
    tax = round(amount * 0.13, 2)
    options = [
        (
            f"invoice_{month}_{index}.pdf",
            f"Invoice number INV-{index:04d}. Customer amount due {amount}.00 USD. "
            f"Subtotal {amount - 120}.00 USD. Tax {tax} USD. Payment due within 10 business days.",
        ),
        (
            f"salary_slip_{month}_{index}.xlsx",
            f"Payroll statement for employee {index}. Base salary {amount}.00 RUB. "
            f"Bonus 5000.00 RUB. Income tax {tax} RUB. Net pay {round(amount - tax + 5000, 2)} RUB.",
        ),
        (
            f"budget_report_q{random.randint(1, 4)}_{index}.csv",
            f"Budget allocation table. Marketing {amount}. Operations {amount // 2}. "
            f"Forecast variance 4.5 percent. Approved by finance controller.",
        ),
    ]
    return random.choice(options)


def build_study_sample(index: int) -> tuple[str, str]:
    options = [
        (
            f"lecture_ml_week_{index}.pdf",
            "Lecture notes on gradient descent, loss minimization, matrix derivatives, "
            "regularization, and overfitting. Formula: J(theta) = 1/m * sum((h(x)-y)^2).",
        ),
        (
            f"homework_algebra_{index}.docx",
            "Homework assignment: solve linear systems, compute determinants, prove theorem, "
            "and derive eigenvalues for the provided matrices.",
        ),
        (
            f"exam_prep_physics_{index}.txt",
            "Study guide covering Newton laws, impulse conservation, oscillations, "
            "laboratory questions, and sample derivations for final exam preparation.",
        ),
    ]
    return random.choice(options)


def build_media_sample(index: int) -> tuple[str, str]:
    extensions = ["jpg", "png", "mp4", "mp3", "gif", "wav", "mkv"]
    stems = ["vacation_clip", "podcast_episode", "thumbnail", "camera_roll", "screen_capture"]
    return f"{random.choice(stems)}_{index}.{random.choice(extensions)}", ""


def build_archives_sample(index: int) -> tuple[str, str]:
    extensions = ["zip", "rar", "7z", "tar", "tar.gz"]
    stems = ["backup_bundle", "project_snapshot", "release_build", "logs_archive", "assets_package"]
    return f"{random.choice(stems)}_{index}.{random.choice(extensions)}", ""


def build_logs_sample(index: int) -> tuple[str, str]:
    status = random.choice(["INFO", "WARN", "ERROR"])
    ip_octet = random.randint(10, 240)
    endpoint = random.choice(["/api/login", "/api/files", "/health", "/billing/charge"])
    return (
        f"service_{index}.log",
        f"2026-03-{random.randint(1, 28):02d} 10:{random.randint(10, 59):02d}:{random.randint(10, 59):02d} "
        f"{status} request_id=req-{index:05d} ip=192.168.{ip_octet}.{random.randint(1, 254)} "
        f"endpoint={endpoint} latency_ms={random.randint(12, 980)} "
        f"message=background worker processed event batch {index}.",
    )


def build_other_sample(index: int) -> tuple[str, str]:
    gibberish = [
        "lorem zqxv packet amber flux nine",
        "tmp raw object delta noodle frame",
        "unsorted bundle item alpha beta gamma",
        "draft final final maybe keep this one",
    ]
    extensions = ["bin", "dat", "tmp", "bak", "blob", "misc"]
    return f"unknown_item_{index}.{random.choice(extensions)}", random.choice(gibberish)


def build_rows() -> list[dict[str, str]]:
    generators = {
        "Code": build_code_sample,
        "Docs": build_docs_sample,
        "Finance": build_finance_sample,
        "Study": build_study_sample,
        "Media": build_media_sample,
        "Archives": build_archives_sample,
        "Logs": build_logs_sample,
        "Other": build_other_sample,
    }

    rows: list[dict[str, str]] = []
    for label, builder in generators.items():
        for index in range(SAMPLES_PER_CLASS):
            filename, content = builder(index)
            rows.append(
                {
                    "filename": filename,
                    "content": content,
                    "label": label,
                }
            )

    random.shuffle(rows)
    return rows


def main() -> None:
    random.seed(RANDOM_SEED)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    df = pd.DataFrame(rows, columns=["filename", "content", "label"])
    df.to_csv(DATASET_PATH, index=False)

    print(f"[OK] Dataset created: {len(df)} rows")
    print(df["label"].value_counts().sort_index())
    print(f"[OK] Saved to: {DATASET_PATH}")


if __name__ == "__main__":
    main()
