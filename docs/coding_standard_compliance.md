# Coding Standard Compliance

_Last updated: 2026-09-14_

The repository follows the internship Python coding standard using a **progressive migration** policy: reusable/new code under `src/` and `tests/` is held to the current standard, while large historical experiment scripts remain reproducibility artifacts and are migrated incrementally.

## Enforced for core code

- Python 3.10 target.
- PEP 8 / Google-style engineering conventions.
- Black-compatible 88-character line width.
- `snake_case` functions/modules and `PascalCase` classes.
- Absolute imports in reusable package code.
- Public modules/classes/functions include docstrings.
- Public function interfaces use type annotations where practical.
- No bare `except:` or silent exception swallowing in new core utilities.
- Unit tests live under `tests/` and use assertions rather than `print`.
- Runtime secrets, datasets, model checkpoints, virtual environments and generated outputs are excluded by `.gitignore`.

## Tooling

Development dependencies include Black, isort, flake8, mypy, pylint and pytest. GitHub Actions checks `src/` and `tests/` and compiles `scripts/` to catch syntax regressions.

## Historical experiment scripts

Several scripts under `scripts/` are longer than the coding-standard recommendation of 500 lines because they preserve the exact experimental workflow used during rapid evaluation, including the frozen v3 verified-positive benchmark. They are treated as **historical/reproducibility entry points**, not as reusable library modules.

The migration rule is:

1. do not rewrite a frozen experiment in a way that changes recorded results;
2. extract stable reusable logic into `src/fashion_multimodal_analysis/`;
3. keep experiment entry points thin in subsequent iterations;
4. require all newly added reusable code to pass the core CI checks.

This follows the coding standard's progressive-migration principle for existing code while keeping new reusable code compliant.
