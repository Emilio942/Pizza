# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code (`pizza_detector.py`, augmentation, utils, emulation, models, api).
- `tests/`: Pytest suite (`test_*.py`, shared fixtures in `conftest.py`).
- `scripts/`: Training, evaluation, augmentation, and utilities.
- `data/`, `models/`: Datasets and artifacts (keep out of commits when possible).
- `config/`: JSON configs used by scripts and modules.
- `docs/`, `hardware/`, `deployment/`: Documentation, hardware files, deployment helpers.

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run detector (train/export): `python src/pizza_detector.py train` | `python src/pizza_detector.py export`
- Start emulator: `python src/emulation/emulator.py`
- Run tests (quiet/stop early): `pytest -q` | `pytest -x -q`
- Coverage: `pytest --cov=src --cov-report=term-missing`

## Coding Style & Naming Conventions
- Python 3.8+, PEP 8, 4‑space indentation, 120‑char soft line limit.
- Naming: modules/functions `lower_snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE` (see `src/constants.py`).
- Type hints for new/modified code; docstrings for public APIs.
- Keep scripts idempotent and path-safe; prefer config in `config/*.json` over hardcoded values.

## Testing Guidelines
- Framework: Pytest (see `tests/`); add tests for new features and bug fixes.
- Naming: `tests/test_<area>.py`, functions `test_<behavior>()`.
- Targeted runs: `pytest tests/test_power_manager.py -k critical`.
- Aim for coverage on changed lines; use `pytest --cov=src` locally before PRs.

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (e.g., `feat: add rp2040 export flag`, `fix: handle empty frame in emulator`).
- PRs must include: clear description, linked issues, test results (paste `pytest` and optional coverage summary), and docs updates when behavior changes.
- Keep diffs focused; include sample commands or paths when relevant.

## Security & Configuration Tips
- Do not commit private data, large binaries, or credentials; use `.env` and `python-dotenv` where needed.
- Place tunables in `config/*.json`; reference via relative paths.
- Validate file I/O and guard long‑running scripts with `if __name__ == "__main__":`.

