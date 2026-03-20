# Contributing to NativeLab

Thanks for your interest in contributing. This document explains how to report bugs, suggest features, and submit code changes.

## Reporting Bugs

Open a GitHub issue and include:

- OS and Python version
- Full error traceback
- Steps to reproduce
- What you expected vs what actually happened

## Suggesting Features

Open an issue before writing any code. Describe what you want and why. This avoids wasted effort if the feature does not fit the project direction.

## Development Setup

```bash
git clone https://github.com/7zonesystems/NativeLab.git
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cd NativeLab
python main.py
```

## Submitting a Pull Request

1. Fork the repository
2. Create a branch from `main`: `git checkout -b your-feature-name`
3. Make your changes
4. Test that the app runs without errors
5. Open a pull request against `main` with a clear description of what changed and why

Do not open PRs directly against `main` without a linked issue or prior discussion.

## Code Style

- Follow PEP8
- Use descriptive variable names
- Keep functions focused on a single responsibility
- No unused imports

## Project-Specific Rules

These exist because of how the codebase is structured. Please follow them to avoid introducing hard-to-debug errors.

**No circular imports.** Do not add top-level imports that create circular dependencies. If a module needs something from a higher-level module, use a lazy import inside the function that needs it:

```python
def some_function():
    from GlobalConfig import config_global
    return config_global.SOME_VALUE
```

**Config values come from GlobalConfig only.** Do not hardcode paths, thread counts, context sizes, or other configuration values. All config goes through `GlobalConfig/config_global.py`.

**New inference engines go in `core/engines/`.** Follow the pattern of existing engines (`llamaengine.py`, `apiengine.py`).

**New UI components go in `UI/Qt6widgets/`.** Keep UI logic separate from business logic.

**New model definitions go in `Model/`.** Do not scatter model-related code across other modules.

## What Not to Contribute

- Features that require an internet connection at runtime (this is a local-first app)
- Dependencies that are not cross-platform
- Large binary files

## Contact

If you have questions, reach out before opening a PR:

- GitHub: [@7zonesystems](https://github.com/7zonesystems)
- Email: 7zonesystems@zohomail.in
