# Repository layout

Zeus is one python package in one repository — unlike [Musibot](https://github.com/OmniOMR/musibot/blob/main/docs/repository-layout.md), which is a monorepo of independently released components. There is one virtual environment, one `pyproject.toml` and one version.

```
zeus/
  src/zeus/          the package
  tests/             unit tests, no weights and no GPU
  docs/              this folder
  .github/           the CI workflow
  datasets/          generated — Zeus datasets and their pickles
  models/            generated — model snapshots
  logs/              generated — training runs, tensorboard, evaluations
  out/               generated — visualizations and predictions
  musescore/         generated — `zeus render` command downloads MuseScore here
```

The four generated folders are gitignored, and are where the CLI writes by default. They are relative to the working directory, so Zeus commands are normally run from the repository root.


## Inside the package

```
src/zeus/
  cli/               one module per `zeus` subcommand, plus the registry
  data/              the Zeus dataset format: samples files, datasets, views
  model/             the model: architecture, weights, options, preprocessing
  musibot/           the worker IPC contract, for running under Musibot
  musicxml/          turning predicted LMX into MusicXML
  evaluation/        metrics
  visualization/     HTML renderings of data and of predictions
```

Two arrangements in there are deliberate and worth knowing.

**`__init__.py` is the public API.** It re-exports the names a library user needs — `Zeus`, the options classes, the dataset classes — and resolves them lazily, so `import zeus` imports nothing and does not pay for TensorFlow. Everything below it is internal arrangement that may move. See [Python API](python-api.md).

**Nothing imports TensorFlow at module level except where it is used.** Every CLI command defers its heavy imports into `execute`, which is what keeps `zeus --help` under two tenths of a second rather than several seconds. `zeus visualize-predictions` and the whole `musibot` package avoid it entirely. A stray top-level `import tensorflow` in a command module would put five seconds in front of every invocation, so [a test](../tests/test_cli.py) guards it.


## Why `src/`

The package sits in `src/` rather than at the repository root, which costs a directory level and means Zeus must be installed — `pip install -e .` — before anything can import it.

It earns that here more than it would in most projects, because Zeus is normally run *from* the repository root: `zeus train` writes to `logs/` relative to the working directory. Without `src/`, a `zeus/` folder at the root would be on `sys.path` for every such invocation, and `import zeus` would resolve to the source tree whether or not anything was installed.

That mostly does not matter in an editable install, where the two are the same code. Where it does matter is packaging. Zeus ships non-python data inside the package — `model/architectures/grand24.yaml` and `solo26.yaml`, loaded relative to `__file__` — and if a build change ever dropped those from the wheel, a flat layout would hide it completely: development runs would read them off disk regardless, tests would pass, and the breakage would appear only for someone installing from the git URL. With `src/`, `import zeus` can only resolve to the installed distribution, so a wheel missing its data files fails for us first.

Musibot's components are flat, so this is a place where the two repositories differ. That is a layout choice rather than a convention the codebases need to share: Musibot's components are libraries installed into service environments and never run from their own directory.


## Tests

`tests/` holds unit tests only. They load no weights, need no GPU, and the whole suite runs in about ten seconds — of which most is two subprocess tests that deliberately import TensorFlow to prove things about the import graph.

That speed is a property worth protecting rather than an accident. The model classes are testable without a model: the Musibot handler takes a `Predictor` protocol, so a stub stands in for Zeus; the visualizers take data rather than a model; and preprocessing, the token map, the metrics and the LMX-to-MusicXML conversion are all pure functions of their inputs.

What is *not* covered is anything requiring trained weights — that a model reads music correctly is a question for evaluation against a dataset, not for a unit test. See [Training Zeus](training-zeus.md).
