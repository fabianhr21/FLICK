# Contributing to FLICK

Thank you for your interest in contributing!

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/fabianhr21/FLICK/issues) with:
- A minimal reproducer (script + input file)
- Python version, OS, and relevant package versions
- Expected vs. actual behaviour

## Feature Requests

Open an issue describing the use case and expected API. Bonus points for a sketch implementation.

## Pull Requests

1. Fork the repository and create a branch from `main`.
2. Make your changes with a clear, focused scope.
3. Add or update tests in `Testsuite/` for every changed behaviour.
4. Verify all tests pass:
   ```bash
   pip install -e ".[dev]"
   pytest Testsuite/ -v
   ```
5. Open a PR with a description of what changed and why.

## Development Setup

```bash
git clone --recurse-submodules https://github.com/fabianhr21/FLICK.git
cd FLICK
pip install -e ".[dev,gpu]"   # add [hpc] if you need mpi4py
pytest Testsuite/ -v
```

City4CFD (for the `geo4cfd` preprocessing stage) must be compiled separately:
```bash
bash scripts/compile_tools.sh
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Add a docstring to every public function and class.
- Keep module-level side effects out of importable files (use `if __name__ == '__main__':` guards).
- All inline comments and docstrings must be in English.

## Questions

Contact: fabian.alexis.hernandez@upc.edu
