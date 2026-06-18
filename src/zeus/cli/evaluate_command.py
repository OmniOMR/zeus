import argparse
from pathlib import Path
from ..model.InferenceOptions import InferenceOptions
from datetime import datetime


def define_parser(parser: argparse.ArgumentParser):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to a trained model folder e.g. 'models/zeus-olimpic-1.0-2024-02-12.model'"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset pickle used for evaluation"
    )
    parser.add_argument(
        "--output",
        default=f"out/evaluation-{timestamp}",
        type=str,
        help=
            "Path to the output folder where evaluation " +
            "results willl be written"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Number of samples per batch when doing inference"
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # deffered imports as they import tensorflow which is slow
    from ..model.Zeus import Zeus
    from ..data.PickledDataset import PickledDataset

    # prepare CLI arguments
    model_folder_path = Path(args.model)
    dataset_pickle_path = Path(args.dataset)
    output_path = Path(args.output)
    batch_size = int(args.batch_size)

    # load the dataset
    dataset = PickledDataset.from_pickle_file(dataset_pickle_path)
    dataset.print_statistics()

    # run model prediction
    zeus = Zeus.load(model_folder_path)
    zeus.evaluate(
        dataset=dataset,
        inference_options=InferenceOptions(
            batch_size=batch_size
        ),
        with_progress_bar=True,
        write_predictions_to=output_path / "predictions.lmx",
        write_metrics_to=output_path / "metrics.yaml",
    )
