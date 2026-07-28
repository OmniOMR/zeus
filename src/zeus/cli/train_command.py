import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from ..data.shuffled_view import ShuffledView
from ..model.architecture_options import ArchitectureOptions
from ..model.inference_options import InferenceOptions
from ..model.model_options import KNOWN_SUBDIVISIONS, ModelOptions
from ..model.token_map import TokenMap
from ..model.training_options import TrainingOptions

NAME = "train"

DESCRIPTION = "Trains a new model on the given dataset"


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--experiment", type=str, required=True, help="Name of the experiment, used in logs"
    )
    parser.add_argument(
        "--model-snapshot",
        default=None,
        type=str,
        help="Path to load a model from to refine instead of "
        + "training a new one, e.g. 'models/zeus-olimpic-1.0-2024-02-12.model'",
    )
    parser.add_argument(
        "--input-subdivisions",
        default=None,
        type=str,
        nargs="+",
        choices=list(KNOWN_SUBDIVISIONS),
        help="Which Musicorpus page subdivisions the trained model will be "
        + "able to read, e.g. 'Staves' for a solo-staff model or "
        + "'Grandstaves' for a piano model. Required when training a new "
        + "model; when fine-tuning, defaults to whatever the loaded snapshot "
        + "declares. This is stored in the snapshot and is what a Musibot "
        + "worker announces, so it decides which images the model is sent.",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        type=str,
        help="When training a new model, this argument specifies its "
        + "architecture. Use 'grand24' for the grand staff model from 2024 "
        + "and 'solo26' for the solo-staff model from 2026.",
    )
    parser.add_argument(
        "--train",
        required=True,
        type=str,
        nargs="*",
        help="Path to the dataset pickle used for training",
    )
    parser.add_argument(
        "--augment",
        default="h:8",
        type=str,
        help="Data augmentation instructions, defaults to 'h:8'",
    )
    parser.add_argument(
        "--dev",
        type=str,
        default=[],
        nargs="*",
        help="Paths to dataset pickles used for validation",
    )
    parser.add_argument(
        "--test", type=str, default=[], nargs="*", help="Paths to dataset pickles used for testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="How many epochs on the training dataset to train for",
    )
    parser.add_argument(
        "--evaluation-from", type=int, default=1, help="Start evaluation with this epoch onward"
    )
    parser.add_argument(
        "--evaluation-each", type=int, default=1, help="Run evaluation each this number of epochs"
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="Number of samples per batch when doing training"
    )
    parser.add_argument(
        "--learning-rate", default=1e-3, type=float, help="Initial learning rate, defaults to 1e-3"
    )
    parser.add_argument(
        "--lr-decay",
        default="cos",
        choices=["none", "cos"],
        help="Type of learning rate decay, defaults to none",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--threads",
        default=0,
        type=int,
        help="Maximum number of threads to use, 0 meaning " + "automatic setting (default)",
    )
    parser.add_argument(
        "--quiet-tf",
        default=False,
        action="store_true",
        help="Set Tensorflow logging to level 2 "
        + "(hides debugging messages, reports only errors)",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # Report only TF errors
    if args.quiet_tf:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # deffered imports as they import tensorflow which is slow
    import tensorflow as tf

    from ..data.zeus_dataset import ZeusDataset
    from ..model.zeus import Zeus

    # prepare CLI arguments
    experiment = str(args.experiment)
    snapshot_path: Path | None = Path(args.model_snapshot) if args.model_snapshot else None
    architecture: str | None = str(args.architecture) if args.architecture else None
    input_subdivisions: list[str] | None = (
        list(args.input_subdivisions) if args.input_subdivisions else None
    )
    train_pickle_paths = [Path(p) for p in args.train]
    augmentations = str(args.augment)
    dev_pickle_paths = [Path(p) for p in args.dev]
    test_pickle_paths = [Path(p) for p in args.test]
    epochs = int(args.epochs)
    evaluation_from = int(args.evaluation_from)
    evaluation_each = int(args.evaluation_each)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    # The parser restricts this to the same two values the Literal names, so
    # the cast asserts what argparse has already enforced.
    lr_decay = cast(Literal["none", "cos"], str(args.lr_decay))
    seed = int(args.seed)
    threads = int(args.threads)

    # either load or create a model
    if architecture is None and snapshot_path is None:
        print("Specify either the --model-snapshot or --architecture arguments,")
        print("i.e, you must either load a model or train a new one.")
        sys.exit(1)

    # A new model has nothing to inherit this from, and getting it wrong is
    # not an error anyone would notice later: the snapshot would simply
    # announce that it reads something it was never trained on.
    if snapshot_path is None and input_subdivisions is None:
        print("Specify --input-subdivisions when training a new model,")
        print("i.e. which page subdivisions it will be able to read.")
        print("Choose from:", ", ".join(KNOWN_SUBDIVISIONS))
        sys.exit(1)

    # create the logdir
    # One `now()` for both stamps, so the logdir and the announced version
    # cannot disagree about when this run started.
    started_at = datetime.now()
    timestamp = started_at.strftime("%y%m%d_%H%M%S")
    logdir_path = Path("logs", f"{experiment}-{timestamp}")
    logdir_path.mkdir(parents=True, exist_ok=True)

    # The identity this run's snapshots announce to Musibot. The name is the
    # experiment, so a fine-tuning run produces a different model rather than a
    # new version of its parent; the version is when the run started, and each
    # snapshot appends its own name to it (see Zeus.snapshot_version).
    run_stamp = started_at.strftime("%Y-%m-%d-%H%M%S")

    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    # load training datasets
    print("Loading train datasets...")
    train_datasets = [ZeusDataset.load_from_pickle_file(path) for path in train_pickle_paths]
    for d in train_datasets:
        d.print_statistics()
    train_dataset = ZeusDataset.combine_multiple(train_datasets)
    print("Combined train dataset: ", end="")
    train_dataset.print_statistics()

    # create a shuffled view of the train dataset
    shuffled_train_dataset = ShuffledView.create_random_for(
        dataset=train_dataset,
        seed=seed,
    )

    # load validation datasets
    print("Loading dev datasets...")
    dev_datasets = [ZeusDataset.load_from_pickle_file(path) for path in dev_pickle_paths]
    for d in dev_datasets:
        d.print_statistics()

    # load test datasets
    print("Loading test datasets...")
    test_datasets = [ZeusDataset.load_from_pickle_file(path) for path in test_pickle_paths]
    for d in test_datasets:
        d.print_statistics()

    print("Done loading datasets.")
    print("")

    # create new or load an existing model
    if snapshot_path is None:
        assert architecture is not None
        assert input_subdivisions is not None
        architecture_options = ArchitectureOptions.from_well_known(architecture)
        zeus = Zeus(
            architecture_options=architecture_options,
            token_map=TokenMap.create_from_dataset(train_dataset.samples),
            model_options=ModelOptions(input_subdivisions=input_subdivisions),
        )
    else:
        zeus = Zeus.load(snapshot_path)
        # Fine-tuning inherits what the snapshot says it reads, unless the
        # run is deliberately changing it.
        if input_subdivisions is not None:
            zeus.model_options = replace(zeus.model_options, input_subdivisions=input_subdivisions)

    # Always this run's own, never the parent's: a fine-tuned model that
    # announced its parent's identity would collide with it in Musibot's
    # registry, and the two would be treated as one model scaled out.
    zeus.model_options = replace(
        zeus.model_options,
        musibot_model_name=experiment,
        musibot_model_version=run_stamp,
    )

    print("[Zeus]: The model reads:", ", ".join(zeus.model_options.input_subdivisions))
    print(f"[Zeus]: Snapshots will announce themselves as {experiment} / {run_stamp}-*")

    # train the new model
    zeus.train(
        shuffled_train_dataset=shuffled_train_dataset,
        dev_datasets=dev_datasets,
        test_datasets=test_datasets,
        training_options=TrainingOptions(
            epochs=epochs,
            evaluation_from=evaluation_from,
            evaluation_each=evaluation_each,
            is_finetuning=snapshot_path is not None,
            augmentations=augmentations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            seed=seed,
        ),
        inference_options_for_evaluation=InferenceOptions(
            batch_size=batch_size,
            transformations=[],
        ),
        logdir_path=logdir_path,
    )
