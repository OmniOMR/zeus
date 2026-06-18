import argparse
from pathlib import Path
from ..model.ArchitectureOptions import ArchitectureOptions
from ..model.TokenMap import TokenMap
import os
from datetime import datetime


def define_parser(parser: argparse.ArgumentParser):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset pickle that was used for evaluation"
    )
    parser.add_argument(
        "--predictions",
        default=f"out/visualization-{timestamp}",
        type=str,
        help=
            "Path to the predictions lmx file, the visualisation html " +
            "will be created next to this file."
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # Report only TF errors
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # deffered imports as they import tensorflow which is slow
    from ..data.ZeusDataset import ZeusDataset
    from ..model.Zeus import Zeus

    # prepare CLI arguments
    dataset_pickle_path = Path(args.dataset)
    predictions_file_path = Path(args.predictions)

    # load the dataset
    dataset = ZeusDataset.load_from_pickle_file(dataset_pickle_path)
    dataset.print_statistics()

    # load predictions
    predictions_lmx: list[str] = []
    with open(predictions_file_path, "r") as file:
        for line in file:
            predictions_lmx.append(line.strip())

    # new dummy model to run the visualization with
    zeus = Zeus(
        architecture_options=ArchitectureOptions(),
        token_map=TokenMap.create_from_dataset(dataset.samples)
    )

    # train the new model
    zeus.visualize_predictions(
        title=predictions_file_path.relative_to(predictions_file_path.parent.parent),
        dataset=dataset,
        predictions_lmx=predictions_lmx,
        output_html_path=predictions_file_path.with_suffix(".html"),
        sample_count=100,
    )
