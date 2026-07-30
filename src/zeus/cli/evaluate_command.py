import argparse
import sys
from datetime import datetime
from pathlib import Path

from ..evaluation.metrics import ALL_METRICS, parse_metric_names
from ..model.inference_options import InferenceOptions

NAME = "evaluate"

DESCRIPTION = "Evaluates a trained model against a given dataset"


def define_parser(parser: argparse.ArgumentParser):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    parser.add_argument(
        "--model-snapshot",
        required=True,
        type=str,
        help="Path to a trained model folder e.g. 'models/zeus-olimpic-1.0-2024-02-12.model'",
    )
    parser.add_argument(
        "--dataset", required=True, type=str, help="Path to the dataset pickle used for evaluation"
    )
    parser.add_argument(
        "--output",
        default=f"out/evaluation-{timestamp}",
        type=str,
        help="Path to the output folder where evaluation " + "results willl be written",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        type=str,
        help="Comma-separated list of metrics to compute, e.g. "
        + "'SER,SERpitchonly'. Defaults to SER alone. "
        + "Available: "
        + ", ".join(ALL_METRICS)
        + ".",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Number of samples per batch when doing inference",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # deffered imports as they import tensorflow which is slow
    from ..data.zeus_dataset import ZeusDataset
    from ..model.zeus import Zeus

    # prepare CLI arguments
    model_folder_path = Path(args.model_snapshot)
    dataset_pickle_path = Path(args.dataset)
    output_path = Path(args.output)
    batch_size = int(args.batch_size)
    try:
        metrics = parse_metric_names(args.metrics) if args.metrics else None
    except ValueError as error:
        print(error)
        sys.exit(2)

    # load the dataset
    dataset = ZeusDataset.load_from_pickle_file(dataset_pickle_path)
    dataset.print_statistics()

    # run model prediction
    zeus = Zeus.load(model_folder_path)
    zeus.evaluate(
        dataset=dataset,
        inference_options=InferenceOptions(batch_size=batch_size),
        with_progress_bar=True,
        metrics=metrics,
        write_predictions_to=output_path / "predictions.lmx",
        write_metrics_to=output_path / "metrics.yaml",
    )
