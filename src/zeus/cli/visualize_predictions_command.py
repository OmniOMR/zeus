import argparse
import sys
from pathlib import Path

NAME = "visualize-predictions"

DESCRIPTION = "Visualizes predictions that result from the 'evaluate' command"


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the dataset pickle that was used for evaluation",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to the predictions lmx file written by 'zeus evaluate'. "
        + "The visualisation html will be created next to this file.",
    )
    parser.add_argument(
        "--sample-count",
        default=100,
        type=int,
        help="How many samples to show, chosen at random, defaults to 100",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # No TensorFlow anywhere in this command: the predictions have already been
    # made, so this only reads files and computes an error rate.
    from ..data.zeus_dataset import ZeusDataset
    from ..visualization.visualize_predictions import visualize_predictions

    # prepare CLI arguments
    dataset_pickle_path = Path(args.dataset)
    predictions_file_path = Path(args.predictions)
    sample_count = int(args.sample_count)

    for path in (dataset_pickle_path, predictions_file_path):
        if not path.is_file():
            print("There is no file at", path)
            sys.exit(2)

    # load the dataset
    dataset = ZeusDataset.load_from_pickle_file(dataset_pickle_path)
    dataset.print_statistics()

    # load predictions
    predictions_lmx: list[str] = []
    with open(predictions_file_path) as file:
        for line in file:
            predictions_lmx.append(line.strip())

    visualize_predictions(
        title=str(predictions_file_path.relative_to(predictions_file_path.parent.parent)),
        dataset=dataset,
        predictions_lmx=predictions_lmx,
        output_html_path=predictions_file_path.with_suffix(".html"),
        sample_count=sample_count,
    )
