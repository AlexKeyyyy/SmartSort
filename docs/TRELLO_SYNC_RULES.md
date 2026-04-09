# Trello Sync Rules

Use these rules to keep board state aligned with repository state.

## 1) Close cards based on evidence

- Move card to `Done` only when there is evidence:
  - commit hash, or
  - concrete file path, or
  - demo screenshot/video

## 2) Assign ownership and due dates

- Every non-done card must have:
  - assignee
  - due date within current week

## 3) Keep comments short and auditable

- Suggested card comment format:
  - `Done in: <file path or commit>`
  - `Checked by: <name>`
  - `Proof: <screen or artifact path>`

## 4) Daily board routine (5-10 minutes)

- Sync in this order:
  - update card status
  - assign owner for remaining cards
  - confirm this week due dates
  - mark blockers explicitly

## 5) Submission week rule

- Before final submission, no open card should be left without owner/date/evidence note.

## 6) Recommended cards to add now

- `Обучающий конспект с нуля`
  - evidence: `docs/PROJECT_FROM_ZERO.md`
- `Шпаргалка для защиты`
  - evidence: `docs/DEFENSE_CHEATSHEET.md`
- `RU datasets shortlist + benchmark`
  - evidence:
    - `docs/RU_DATASETS_SHORTLIST.md`
    - `docs/RU_DATASET_BENCHMARK_PROTOCOL.md`
    - `data/open_dataset_manifest.ru_extended.json`
    - `reports/model_eval/comparison_ru_extended.md` (after benchmark run)
