import argparse
from pathlib import Path
from ..model.ArchitectureOptions import ArchitectureOptions
from ..model.TrainingOptions import TrainingOptions
from ..model.TokenMap import TokenMap
import os
from datetime import datetime
from ..data.ZeusDataset import ZeusDataset
from ..data.ShuffledView import ShuffledView


def define_parser(parser: argparse.ArgumentParser):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    parser.add_argument(
        "--train",
        required=True,
        type=str,
        nargs="*",
        help="Path to the dataset pickle used for training"
    )
    parser.add_argument(
        "--augment",
        default="h:8",
        type=str,
        help="Data augmentation instructions, defaults to 'h:8'"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Number of samples per batch when doing training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed"
    )
    parser.add_argument(
        "--output",
        default=f"out/visualization-{timestamp}",
        type=str,
        help="Path to a folder that will contain the visualisation files"
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # Report only TF errors
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # deffered import since it imports tensorflow which is slow
    from ..model.Zeus import Zeus

    # prepare CLI arguments
    train_pickle_paths = [Path(p) for p in args.train]
    augmentations = str(args.augment)
    batch_size = int(args.batch_size)
    seed = int(args.seed)
    output_path = Path(args.output)

    # load training datasets
    train_datasets = [
        ZeusDataset.load_from_pickle_file(path)
        for path in train_pickle_paths
    ]
    for d in train_datasets: d.print_statistics()
    train_dataset = ZeusDataset.combine_multiple(train_datasets)
    train_dataset.print_statistics()

    # create a shuffled view of the train dataset
    shuffled_train_dataset = ShuffledView.create_random_for(
        dataset=train_dataset,
        seed=seed,
    )

    # new dummy model to run the visualization with
    zeus = Zeus(
        architecture_options=ArchitectureOptions(),
        token_map=TokenMap.create_from_dataset(train_dataset.samples)
    )

    # train the new model
    zeus.visualize_training_data(
        shuffled_train_dataset=shuffled_train_dataset,
        training_options=TrainingOptions(
            epochs=0,
            evaluation_from=0,
            evaluation_each=0,
            is_finetuning=False,
            augmentations=augmentations,
            batch_size=batch_size,
        ),
        output_folder_path=output_path,
    )
