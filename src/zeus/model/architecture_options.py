import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ArchitectureOptions:
    """
    Parameters for Zeus architecture. Defines a specific
    architecture of the model - specific sizes, dimensions
    and layer counts.
    """

    name: str
    """Human-readable name of the architecture, e.g. 'grand24'"""

    height: int
    """Image height."""

    cnn_dim: int
    """CNN dim at original resolution."""

    cnn_resblocks: int
    """CNN ResNet blocks per layer."""

    cnn_stages: int
    """CNN layers."""

    rnn_dim: int
    """RNN dimension."""

    rnn_layers: int
    """RNN layers."""

    rnn_layers_decoder: int
    """RNN decoder layers."""

    timestep_width: int
    """Timestep width."""

    dropout: float
    """What dropout rate should be used during model training"""

    @staticmethod
    def from_well_known(architecture_name: str) -> "ArchitectureOptions":
        archs_path = Path(__file__).parent / "architectures"

        if architecture_name == "grand24":
            return ArchitectureOptions.from_yaml(archs_path / "grand24.yaml")
        elif architecture_name == "solo26":
            return ArchitectureOptions.from_yaml(archs_path / "solo26.yaml")

        raise ValueError("Unknown model architecture: " + architecture_name)

    @staticmethod
    def from_model_folder(model_folder_path: Path) -> "ArchitectureOptions":
        """Loads options from the folder of a trained model."""
        # primarily load from the YAML file
        yaml_path = model_folder_path / "architecture_options.yaml"
        if yaml_path.exists():
            return ArchitectureOptions.from_yaml(yaml_path)

        # this is a fallback for old models where YAML is missing
        return ArchitectureOptions.from_legacy_json(model_folder_path / "options.json")

    def write_to_model_folder(self, model_folder_path: Path):
        """Writes our architecture options into a model folder"""
        self.write_to_yaml_file(model_folder_path / "architecture_options.yaml")

    @staticmethod
    def from_yaml(file_path: Path) -> "ArchitectureOptions":
        """
        Loads options from `architecture_options.yaml`
        file that is part of a trained model folder.
        """
        with open(file_path) as file:
            yaml_data: dict = yaml.safe_load(file)
            return ArchitectureOptions(**yaml_data)

    def write_to_yaml_file(self, file_path: Path):
        """Writes our architecture options into a yaml file"""
        yaml_data: dict = asdict(self)
        with open(file_path, "w") as file:
            yaml.dump(yaml_data, file)

    @staticmethod
    def from_legacy_json(json_file_path: Path) -> "ArchitectureOptions":
        """
        Loads options from the legacy `options.json` file that
        the old zeus models use. The old JSON file is just a dump of
        the argparse namespace values.
        """
        options: dict[str, Any] = json.loads(json_file_path.read_text())
        return ArchitectureOptions(
            name=str(options.get("name", "")),  # optional, because introduced later
            height=int(options["height"]),
            cnn_dim=int(options["cnn_dim"]),
            cnn_resblocks=int(options["cnn_resblocks"]),
            cnn_stages=int(options["cnn_stages"]),
            rnn_dim=int(options["rnn_dim"]),
            rnn_layers=int(options["rnn_layers"]),
            rnn_layers_decoder=int(options["rnn_layers_decoder"]),
            timestep_width=int(options["timestep_width"]),
            dropout=float(options["dropout"]),
        )
