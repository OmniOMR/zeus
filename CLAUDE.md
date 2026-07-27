# Zeus

Zeus is a deep learning model for Optical Music Recognition (OMR): it reads an image of a staff or a grandstaff and produces the music notation on it, as LMX tokens and from those as MusicXML. The repository is a single python package that trains the model, stores and loads it as snapshots, and runs it for inference. See [README.md](README.md) and the [docs/](docs/) folder.

Zeus is also deployed as a *Model* inside [Musibot](https://github.com/OmniOMR/musibot), which runs it as a subprocess over a pipe protocol. Musibot's conventions are the ones this repository follows where the two overlap.


## Python version

This package requires **python 3.10** and cannot run on anything newer, because TensorFlow 2.12 has no wheels for 3.11+. That constraint is load-bearing rather than incidental: it is why a Musibot *Worker Head* (which needs 3.11+) has to run Zeus in a virtual environment of its own, across a process boundary.


## Toolchain

The development environment is `.venv`, created with python 3.10 and installed with `.venv/bin/pip install -e '.[dev]'`. Everything runs out of it:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

mypy is deliberately **not** `strict` here, unlike the Musibot components. TensorFlow and h5py ship no type information and this codebase descends from research code, so strict mode would report hundreds of errors that say nothing about correctness. The configuration in `pyproject.toml` is the contract; tighten it as annotations arrive rather than adding `# type: ignore`.

Tests must not import TensorFlow unless they genuinely exercise it — the suite is expected to run in well under a second, which is what makes it worth running.


## Markdown conventions

Match the repository's existing markdown style when creating or editing `.md` files:

- Leave exactly one blank line after a heading.
- Leave exactly two blank lines before a heading (unless the heading is the first line of the file).
- Do not hard-wrap paragraphs — write each paragraph as a single line and let the editor soft-wrap it. Explicit wrapping leaves stray single words on their own lines in the author's editor.
