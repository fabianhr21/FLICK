# Building the Documentation

## Requirements

```bash
pip install -e .[docs]
```

## Build

```bash
sphinx-build -b html docs/source docs/build
```

Open `docs/build/index.html` in a browser.

## Check for warnings (CI mode)

```bash
sphinx-build -b html docs/source docs/build -W
```
