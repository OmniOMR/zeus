import argparse
from pathlib import Path
from ..model.ArchitectureOptions import ArchitectureOptions
from ..model.TrainingOptions import TrainingOptions
from ..model.InferenceOptions import InferenceOptions
from ..model.TokenMap import TokenMap
from ..data.ShuffledView import ShuffledView
from datetime import datetime
import os


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment, used in logs"
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help=
            "Path to load a model from to refine instead of " +
            "training a new one, e.g. 'models/zeus-olimpic-1.0-2024-02-12.model'"
    )
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
        "--dev",
        type=str,
        default=[],
        nargs="*",
        help="Paths to dataset pickles used for validation"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=[],
        nargs="*",
        help="Paths to dataset pickles used for testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="How many epochs on the training dataset to train for"
    )
    parser.add_argument(
        "--evaluation_from",
        type=int,
        default=1,
        help="Start evaluation with this epoch onward"
    )
    parser.add_argument(
        "--evaluation_each",
        type=int,
        default=1,
        help="Run evaluation each this number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Number of samples per batch when doing training"
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Initial learning rate, defaults to 1e-3"
    )
    parser.add_argument(
        "--lr_decay",
        default="cos",
        choices=["none", "cos"],
        help="Type of learning rate decay, defaults to none"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed"
    )
    parser.add_argument(
        "--threads",
        default=0,
        type=int,
        help="Maximum number of threads to use, 0 meaning " +
            "automatic setting (default)"
    )
    parser.add_argument(
        "--quiet_tf",
        default=False,
        action="store_true",
        help=
            "Set Tensorflow logging to level 2 " +
            "(hides debugging messages, reports only errors)"
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    # Report only TF errors
    if args.quiet_tf:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # deffered imports as they import tensorflow which is slow
    from ..data.ZeusDataset import ZeusDataset
    from ..model.Zeus import Zeus
    import tensorflow as tf

    # prepare CLI arguments
    experiment = str(args.experiment)
    model_path: Path | None = Path(args.model) if args.model else None
    train_pickle_paths = [Path(p) for p in args.train]
    augmentations = str(args.augment)
    dev_pickle_paths = [Path(p) for p in args.dev]
    test_pickle_paths = [Path(p) for p in args.test]
    epochs = int(args.epochs)
    evaluation_from = int(args.evaluation_from)
    evaluation_each = int(args.evaluation_each)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    lr_decay = str(args.lr_decay)
    seed = int(args.seed)
    threads = int(args.threads)

    # create the logdir
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    logdir_path = Path("logs", f"{experiment}-{timestamp}")
    logdir_path.mkdir(parents=True, exist_ok=True)

    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    # load training datasets
    train_datasets = [
        ZeusDataset.load_from_pickle_file(path)
        for path in train_pickle_paths
    ]
    for d in train_datasets: d.print_statistics()
    train_dataset = ZeusDataset.combine_multiple(train_datasets)
    print("Combined train dataset: ", end="")
    train_dataset.print_statistics()

    # create a shuffled view of the train dataset
    shuffled_train_dataset = ShuffledView.create_random_for(
        dataset=train_dataset,
        seed=seed,
    )

    # load validation datasets
    dev_datasets = [
        ZeusDataset.load_from_pickle_file(path)
        for path in dev_pickle_paths
    ]
    for d in dev_datasets: d.print_statistics()

    # load test datasets
    test_datasets = [
        ZeusDataset.load_from_pickle_file(path)
        for path in test_pickle_paths
    ]
    for d in test_datasets: d.print_statistics()

    # create new or load an existing model
    if model_path is None:
        zeus = Zeus(
            architecture_options=ArchitectureOptions(),
            token_map=TokenMap.create_from_dataset(train_dataset.samples)
        )
    else:
        zeus = Zeus.load(model_path)

    # train the new model
    zeus.train(
        shuffled_train_dataset=shuffled_train_dataset,
        dev_datasets=dev_datasets,
        test_datasets=test_datasets,
        training_options=TrainingOptions(
            epochs=epochs,
            evaluation_from=evaluation_from,
            evaluation_each=evaluation_each,
            is_finetuning=model_path is not None,
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
        logdir_path=logdir_path
    )
