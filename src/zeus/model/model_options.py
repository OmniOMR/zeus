"""What a model reads, as opposed to how it computes.

`ArchitectureOptions` defines the computation graph — two models with equal
architecture options have identical graphs. This holds the other kind of fact
about a snapshot: properties that do not shape a single tensor but that
somebody deploying the model has to know.

Today that is one thing, the kind of image the model was trained to read, and
it is here rather than in the architecture because it changes nothing about the
graph. A solo-staff model and a grandstaff model can share an architecture
exactly; what differs is what you may hand them.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

KNOWN_SUBDIVISIONS = ("Grandstaves", "Staves", "Systems")
"""The page subdivisions the Musicorpus Specification defines. `Staves` for
solo-staff models, `Grandstaves` for piano models, `Systems` for models reading
a whole system at once."""

DEFAULT_INPUT_SUBDIVISIONS = ["Grandstaves"]
"""What a snapshot written before this file existed is assumed to read.

Every Zeus model up to now was a grandstaff model bar one, so this is right far
more often than not — and a snapshot that disagrees can be corrected by writing
the file into its folder by hand, without retraining anything.
"""

DEFAULT_MUSIBOT_MODEL_NAME = "zeus"
"""What a snapshot that does not name itself is announced as."""


@dataclass
class ModelOptions:
    """Properties of a trained model that are not properties of its graph."""

    input_subdivisions: list[str] = field(default_factory=lambda: list(DEFAULT_INPUT_SUBDIVISIONS))
    """Which Musicorpus page subdivisions this model can read, e.g.
    `["Staves"]` for a solo-staff model, `["Grandstaves"]` for a piano model,
    or both for a model trained on both. Order does not matter: this is a set
    written as a list, and it is normalized to a canonical order on load.

    This is what a Musibot worker announces as the input half of its signature,
    which makes it a wire contract rather than a note to the reader — so it
    travels inside the snapshot, where it cannot be got wrong at deployment
    time. A grandstaff model told to read staves does not fail; it transcribes
    confidently and wrongly, and nothing anywhere notices.
    """

    musibot_model_name: str | None = None
    """The name a Musibot worker announces for this snapshot, which is what a
    *Pipeline* pins. Set from the training run's `--experiment` name, so a
    fine-tuned model is a different model rather than a new version of its
    parent. `None` falls back to `zeus`.

    Prefixed because these two are deployment identity rather than anything
    Zeus itself reads, and because `ArchitectureOptions` already has a `name`
    holding `grand24` or `solo26` — one file away, meaning something else.
    """

    musibot_model_version: str | None = None
    """The version announced beside the name, which together identify the
    model to Musibot. Set from the training run's start time and the snapshot
    within that run, e.g. `2026-07-28-143052-e50`.

    The snapshot's own name has to be in there. `store` is called once per
    evaluated epoch, so every epoch of one run shares a start time, and
    choosing the best epoch off the validation curve is the documented
    workflow — without it, deploying `e40` and `e50` would announce the same
    identity and Musibot would merge them into one registry entry.

    `None` falls back to the snapshot folder's own name.
    """

    def __post_init__(self) -> None:
        if not self.input_subdivisions:
            raise ValueError(
                "A model must read at least one subdivision; "
                f"choose from {', '.join(KNOWN_SUBDIVISIONS)}."
            )

        unknown = [
            subdivision
            for subdivision in self.input_subdivisions
            if subdivision not in KNOWN_SUBDIVISIONS
        ]
        if unknown:
            raise ValueError(
                f"Unknown page subdivision(s) {', '.join(repr(u) for u in unknown)}. "
                f"The Musicorpus Specification defines {', '.join(KNOWN_SUBDIVISIONS)}."
            )

        # Deduplicated and sorted, so that this behaves as the set it is: two
        # snapshots listing the same subdivisions compare equal however they
        # were written, and a worker's announcement is byte-for-byte stable.
        self.input_subdivisions = sorted(set(self.input_subdivisions))

    def accepts(self, subdivision: str) -> bool:
        """Whether this model can read the given Musicorpus subdivision."""
        return subdivision in self.input_subdivisions

    def musibot_identity(self, model_folder_path: Path) -> tuple[str, str]:
        """The name and version a Musibot worker announces for this snapshot.

        Together these are the model's identity to Musibot, and two workers
        announcing the same pair are taken to be the same model scaled
        horizontally. Snapshots that do not name themselves — every one written
        before these fields existed — fall back to `zeus` and to the snapshot
        folder's own name, which in practice is descriptive and unique:
        `zeus-olimpic-1.0-2024-02-12.model` announces that as its version.
        """
        return (
            self.musibot_model_name or DEFAULT_MUSIBOT_MODEL_NAME,
            self.musibot_model_version or model_folder_path.name.removesuffix(".model"),
        )

    @staticmethod
    def from_model_folder(model_folder_path: Path) -> "ModelOptions":
        """Loads options from the folder of a trained model.

        Snapshots written before this file existed simply do not have it, and
        are read as the defaults rather than refused.
        """
        yaml_path = model_folder_path / "model_options.yaml"
        if yaml_path.exists():
            return ModelOptions.from_yaml(yaml_path)

        return ModelOptions()

    def write_to_model_folder(self, model_folder_path: Path) -> None:
        """Writes our model options into a model folder"""
        self.write_to_yaml_file(model_folder_path / "model_options.yaml")

    @staticmethod
    def from_yaml(file_path: Path) -> "ModelOptions":
        """Loads options from a `model_options.yaml` file"""
        with open(file_path) as file:
            yaml_data: dict = yaml.safe_load(file)
            return ModelOptions(**yaml_data)

    def write_to_yaml_file(self, file_path: Path) -> None:
        """Writes our model options into a yaml file"""
        yaml_data: dict = asdict(self)
        with open(file_path, "w") as file:
            yaml.dump(yaml_data, file)
