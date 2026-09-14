# Repository Maintenance Notes

## What belongs in Git

- reusable source code under `src/`;
- experiment entry points under `scripts/`;
- configuration files under `configs/`;
- structured summaries under `reports/`;
- maintained Markdown documentation under `docs/`;
- small tests.

## What should stay local / external

- `.venv` and other virtual environments;
- full DeepFashion2 data;
- Hugging Face caches / model weights;
- `outputs/` image-heavy experiment results;
- temporary GPU transfer bundles;
- generated zip files.

## Documentation rule

Every maintained Markdown file should use GitHub-compatible headings, tables and fenced code blocks. Early pilot metrics must be labelled as pilot prediction coverage if they have not been manually validated as localization accuracy.
