import argparse
import os
from datetime import datetime
from pathlib import Path

from ..model.architecture_options import ArchitectureOptions
from ..model.token_map import TokenMap

NAME = "visualize-predictions"

DESCRIPTION = "Visualizes predictions that result from the 'evaluate' command"


def define_parser(parser: argparse.ArgumentParser):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    parser.add_argument(
        "--architecture",
        required=True,
        type=str,
        help="When training a new model, this argument specifies its "
        + "architecture. Use 'grand24' for the grand staff model from 2024 "
        + "and 'solo26' for the solo-staff model from 2026.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset pickle that was used for evaluation",
    )
    parser.add_argument(
        "--predictions",
        default=f"out/visualization-{timestamp}",
        type=str,
        help="Path to the predictions lmx file, the visualisation html "
        + "will be created next to this file.",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # Report only TF errors
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # deffered imports as they import tensorflow which is slow
    from ..data.zeus_dataset import ZeusDataset
    from ..model.zeus import Zeus

    # prepare CLI arguments
    # Required by the parser, so it is always present.
    architecture = str(args.architecture)
    dataset_pickle_path = Path(args.dataset)
    predictions_file_path = Path(args.predictions)

    # load the dataset
    dataset = ZeusDataset.load_from_pickle_file(dataset_pickle_path)
    dataset.print_statistics()

    # load predictions
    predictions_lmx: list[str] = []
    with open(predictions_file_path) as file:
        for line in file:
            predictions_lmx.append(line.strip())

    # new dummy model to run the visualization with
    # TODO: visualization should be extracted out from the Zeus class
    zeus = Zeus(
        architecture_options=ArchitectureOptions.from_well_known(architecture),
        token_map=TokenMap.create_from_dataset(dataset.samples),
    )

    # train the new model
    zeus.visualize_predictions(
        title=str(predictions_file_path.relative_to(predictions_file_path.parent.parent)),
        dataset=dataset,
        predictions_lmx=predictions_lmx,
        output_html_path=predictions_file_path.with_suffix(".html"),
        sample_count=100,
    )
