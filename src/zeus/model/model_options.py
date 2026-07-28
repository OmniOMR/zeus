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
