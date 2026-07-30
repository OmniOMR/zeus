# Development

Clone this repo, create a venv, install the package into it and activate the venv to get the `zeus` CLI command.

This code requires `Python 3.10.7`, because the old version of TensorFlow needs it.

```bash
# clone
git clone git@github.com:OmniOMR/zeus.git
cd zeus

# make venv
python3.10 -m venv .venv

# install itself
.venv/bin/pip3 install -e .

# activate venv
source .venv/bin/activate

# now you can run CLI commands with the local code
zeus --help
```

## Before committing

Run these checks:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

All four run in about fifteen seconds together, which is what makes running them a habit rather than a chore. Keep it that way: the test suite loads no weights and needs no GPU, and the two tests that deliberately import TensorFlow do it in a subprocess so the rest of the suite does not pay for it.


## Continuous integration

[`.github/workflows/checks.yml`](../.github/workflows/checks.yml) runs exactly those four commands on every push to `main`, every pull request, and every `v*` tag — so what gets released is what was checked.

A second job builds a wheel and looks inside it. That is not redundant with the tests: the package lives in `src/` precisely so that an import can only resolve to the installed distribution, and Zeus ships two architecture YAMLs loaded relative to `__file__`. A wheel that lost them would pass every other check and fail only for somebody installing from the git URL. See [Repository layout](repository-layout.md#why-src).

One line in that workflow is load-bearing and easy to delete by accident:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

The version is derived from `git describe`, and the default checkout is shallow with no tags. Without this Zeus builds as some `0.1.devN` even on a release tag — quietly, since nothing errors. See [Versioning and releases](versioning-and-releases.md#when-the-build-needs-git).
