"""Zeus — an image-to-sequence model for Optical Music Recognition.

Zeus reads an image of a staff or a grandstaff and produces the music notation
on it, as LMX tokens and from those as MusicXML. The package is both a library
and the `zeus` command line tool; see README.md and the docs/ folder.

This module is the public API. Everything a library user needs is reachable
from here::

    from zeus import Zeus, InferenceOptions

    model = Zeus.load(Path("models/solo26.model"))
    lmx = model.predict(images, InferenceOptions())

Module paths below this one are internal arrangement and may move between
versions; names re-exported here will not move without a version bump.


Why these imports are lazy
--------------------------

Importing `Zeus` imports TensorFlow, which takes seconds. Writing the imports
plainly here would make every `import zeus` pay that — including the CLI's own
startup, where `zeus --help` would sit for five seconds before printing a page
of text, and including `import zeus` by anyone who only wanted a dataclass.

So the names are resolved on first access through a module-level `__getattr__`
(PEP 562) and cached in the module's globals thereafter. `from zeus import
Zeus` works exactly as it reads; nothing is imported until the name is actually
touched. The `TYPE_CHECKING` block above gives type checkers and editors the
real definitions, since they never run `__getattr__`.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zeus.data.shuffled_view import ShuffledView
    from zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample
    from zeus.model.architecture_options import ArchitectureOptions
    from zeus.model.inference_options import InferenceOptions
    from zeus.model.token_map import TokenMap
    from zeus.model.training_options import TrainingOptions
    from zeus.model.zeus import Zeus

__all__ = [
    "ArchitectureOptions",
    "InferenceOptions",
    "ShuffledView",
    "TokenMap",
    "TrainingOptions",
    "Zeus",
    "ZeusDataset",
    "ZeusDatasetSample",
]

_EXPORTS = {
    "ArchitectureOptions": "zeus.model.architecture_options",
    "InferenceOptions": "zeus.model.inference_options",
    "ShuffledView": "zeus.data.shuffled_view",
    "TokenMap": "zeus.model.token_map",
    "TrainingOptions": "zeus.model.training_options",
    "Zeus": "zeus.model.zeus",
    "ZeusDataset": "zeus.data.zeus_dataset",
    "ZeusDatasetSample": "zeus.data.zeus_dataset",
}
"""Where each exported name actually lives. Kept beside `__all__` rather than
derived from it, because the two answer different questions: `__all__` is the
promise, this is the lookup."""


def __getattr__(name: str) -> Any:
    """Resolve one exported name, importing its module the first time."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module_name), name)

    # Cached in the module's own globals, so this runs once per name and every
    # later access is an ordinary attribute lookup.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Make the lazy names discoverable to `dir()` and to tab completion."""
    return sorted(__all__)
