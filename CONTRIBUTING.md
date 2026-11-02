# Contributing

Contributions are welcome and appreciated.

## Getting started

```bash
git clone https://github.com/ParraLab-ESBG/pytorch-frustratometer.git
cd pytorch-frustratometer
pip install -e ".[plots]"
```

## Reporting bugs

Open an issue on GitHub with:
- Python and package version (`pip show pytorch-frustratometer`)
- A minimal reproducible example
- Full traceback

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Add or update tests where relevant.
3. Ensure existing tests pass: `pytest tests/`
4. Open a pull request with a clear description of what changed and why.

## Code style

- Follow existing conventions in the file you are editing.
- Keep functions focused; avoid adding abstractions beyond what the change requires.
- Default to writing no comments — only add one when the *why* is non-obvious.

## License

By contributing you agree that your work will be released under the [CC BY 4.0](LICENSE) licence.
