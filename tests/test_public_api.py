"""Tests for the package's public API surface.

`zeus/__init__.py` re-exports its names lazily, which buys `import zeus` its
speed but means nothing checks the export table until someone touches a name.
These tests are that check.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

import zeus

# Everything except `Zeus` itself, which is the one export that pulls in
# TensorFlow. It is covered by its own test below, in a subprocess.
LIGHTWEIGHT_EXPORTS = [name for name in zeus.__all__ if name != "Zeus"]


def test_the_export_table_covers_exactly_what_is_promised() -> None:
    assert sorted(zeus._EXPORTS) == sorted(zeus.__all__)


@pytest.mark.parametrize("name", sorted(zeus.__all__))
def test_every_exported_name_names_a_real_module(name: str) -> None:
    """Catches a typo in the export table without importing anything."""
    assert importlib.util.find_spec(zeus._EXPORTS[name]) is not None


@pytest.mark.parametrize("name", LIGHTWEIGHT_EXPORTS)
def test_exported_names_resolve(name: str) -> None:
    assert getattr(zeus, name).__name__ == name


def test_an_unknown_name_raises_attribute_error() -> None:
    """`__getattr__` must not turn a typo into something stranger."""
    with pytest.raises(AttributeError, match="has no attribute 'Nonesuch'"):
        getattr(zeus, "Nonesuch")  # noqa: B009 — attribute access is the point


def test_the_exports_are_discoverable() -> None:
    """Without `__dir__`, lazy names are invisible to tab completion."""
    assert set(zeus.__all__) <= set(dir(zeus))


def test_importing_the_package_does_not_import_tensorflow() -> None:
    """The whole point of the lazy exports.

    Run in a subprocess because TensorFlow may already be in this process's
    `sys.modules`, put there by some other test, which would make the check
    pass or fail for reasons unrelated to what it is testing.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, zeus; "
            "assert 'tensorflow' not in sys.modules, 'importing zeus imported tensorflow'",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_the_model_is_reachable_from_the_package_root() -> None:
    """`from zeus import Zeus` — the import this whole module exists for.

    In a subprocess, and slow, because resolving this name is what imports
    TensorFlow; keeping it out of process keeps that cost out of every other
    test in the suite.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from zeus import Zeus; assert Zeus.__name__ == 'Zeus'",
        ],
        capture_output=True,
        text=True,
        # Inherited rather than replaced, so the subprocess finds the same
        # installation this process is running out of; only TensorFlow's
        # startup banner is silenced.
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
    )

    assert result.returncode == 0, result.stderr
