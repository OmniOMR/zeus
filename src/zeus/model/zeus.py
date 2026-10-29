import contextlib
import shutil
from dataclasses import replace
from pathlib import Path

import tensorflow as tf
import yaml

from ..data.shuffled_view import ShuffledView
from ..data.zeus_dataset import ZeusDataset
from ..evaluation.metrics import DEFAULT_METRICS, SER, Metric, compute_metrics
from .architecture_options import ArchitectureOptions
from .construct_tf_dataset import construct_tf_dataset, construct_tf_dataset_for_images
from .early_stopping import EarlyStopping, is_evaluation_tick, snapshots_to_collect
from .inference_options import InferenceOptions
from .keras_model import KerasModel
from .model_options import ModelOptions
from .token_map import TokenMap
from .training_options import TrainingOptions

EARLY_STOPPING_EPOCH_LIMIT = 1_000_000
"""What keras is told the length of an early-stopped run is. It needs a number,
and training is ended by the callback long before this one is reached."""


def write_early_stopping_record(file_path: Path, early_stopping: EarlyStopping, dataset_name: str):
    """Writes how early stopping stands, rewritten after every epoch so that a
    run killed partway still says which of its snapshots is the best."""
    with open(file_path, "w") as file:
        yaml.dump(
            {
                "dataset": dataset_name,
                "metric": SER.name,
                "patience": early_stopping.patience,
                "best_epoch": early_stopping.best_epoch,
                "best_value": early_stopping.best_ser,
                "history": early_stopping.history,
            },
            file,
            sort_keys=False,
        )


class Zeus:
    """
    The Zeus model - weigths and architecture.
    Can be trained or used for inference.
    """

    def __init__(
        self,
        architecture_options: ArchitectureOptions,
        token_map: TokenMap,
        model_options: ModelOptions | None = None,
    ):
        """Creates a fresh, initialized and untrained model instance."""

        self.architecture_options = architecture_options
        """Defines concrete sizes, dimensions, and layer counts"""

        self.token_map = token_map
        """Defines the mapping between model outputs and LMX tokens"""

        self.model_options = model_options if model_options is not None else ModelOptions()
        """Says what this model reads, as opposed to how it computes"""

        strategy_scope = (
            tf.distribute.MirroredStrategy().scope()
            if len(tf.config.list_physical_devices("GPU")) > 1
            else contextlib.nullcontext()
        )
        with strategy_scope:
            model = KerasModel(architecture_options=architecture_options, token_map=token_map)

        self.model: KerasModel = model
        """The Keras model containing all the low-level tensorflow stuff"""

    def materialize_weights(self):
        """Run one dummy inference, so that every layer's weights exist.

        Keras builds a layer's weights lazily, on its first call. The encoder is
        a functional model and so is built at construction, but the decoder's
        embedding, RNN cells and output layer are not — until something has been
        decoded, those weights do not exist to be saved or loaded.

        Both directions need this. Loading needs somewhere to put the weights;
        saving needs them to exist, or `save_weights` writes a file with the
        encoder alone in it and the failure surfaces much later, as a layer
        count mismatch, when someone tries to load it.
        """
        self.model.decoder_inference(
            encoded=self.model.encoder(
                tf.RaggedTensor.from_tensor(
                    tf.ones([1, self.architecture_options.height, 128, 1], dtype=tf.float32),
                    ragged_rank=2,
                )
            ),
            max_length=1,
        )

    @staticmethod
    def load(model_folder_path: Path) -> "Zeus":
        """Loads a model from its folder"""
        architecture_options = ArchitectureOptions.from_model_folder(model_folder_path)
        token_map = TokenMap.load_from_model_folder(model_folder_path)
        model_options = ModelOptions.from_model_folder(model_folder_path)
        zeus = Zeus(
            architecture_options=architecture_options,
            token_map=token_map,
            model_options=model_options,
        )

        zeus.load_weights(model_folder_path)

        print("[Zeus]: Loaded model", model_folder_path)

        return zeus

    def load_weights(self, model_folder_path: Path):
        """Replaces this model's weights with those stored in a snapshot folder.

        The snapshot's architecture and token map are not read; they must be
        the ones this model already has.
        """
        self.materialize_weights()

        # keras Layer property, prevents Keras
        # from calling "build" on invocation
        # (and thus overwriting loaded weights)
        # https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/layers/Layer
        # https://github.com/keras-team/keras/blob/v2.12.0/keras/engine/base_layer.py#L364
        self.model.built = True

        # load weights
        self.model.load_weights(str(model_folder_path / "weights.h5"))

    def snapshot_version(self, snapshot_name: str) -> str | None:
        """Compose the announced version of one snapshot of a training run.

        During training `musibot_model_version` holds the run's stamp — the
        moment training started — and each snapshot it produces appends its own
        name to that, so `e40` and `e50` of one run are distinguishable to
        Musibot. Returns None when the run has no stamp, which is what leaves
        the loader falling back to the snapshot folder's name.
        """
        run_stamp = self.model_options.musibot_model_version
        if run_stamp is None:
            return None
        return f"{run_stamp}-{snapshot_name}"

    def store(
        self,
        model_folder_path: Path,
        overwrite: bool = False,
        musibot_model_version: str | None = None,
    ):
        """Stores the model weights and parameters into a folder

        :param musibot_model_version: Written into the snapshot in place of the
            version currently held, without changing this model's own. Training
            uses it to give each epoch's snapshot a version of its own while
            keeping one run-wide stamp in hand for the next.
        """
        if model_folder_path.exists():
            if overwrite:
                shutil.rmtree(model_folder_path)
            else:
                raise Exception("Cannot store model, folder already exists.")

        # create the target folder
        model_folder_path.mkdir(parents=True, exist_ok=True)

        # A model that has been trained has run its decoder and so has all its
        # weights already; one that has only been constructed has not, and
        # saving it would write the encoder alone. Cheap either way, and it
        # makes `store` then `load` a round trip from any state.
        self.materialize_weights()

        # model weights
        self.model.save_weights(str(model_folder_path / "weights.h5"))

        # architecture options
        self.architecture_options.write_to_model_folder(model_folder_path)

        # model options
        model_options = self.model_options
        if musibot_model_version is not None:
            model_options = replace(model_options, musibot_model_version=musibot_model_version)
        model_options.write_to_model_folder(model_folder_path)

        # token map
        self.token_map.write_to_model_folder(model_folder_path)

        print("[Zeus]: Stored model", model_folder_path)

    def train(
        self,
        shuffled_train_dataset: ShuffledView,
        dev_datasets: list[ZeusDataset],
        test_datasets: list[ZeusDataset],
        training_options: TrainingOptions,
        inference_options_for_evaluation: InferenceOptions,
        logdir_path: Path,
        metrics: list[Metric] | None = None,
        early_stopping_dataset: ZeusDataset | None = None,
    ):
        """
        Runs a training procedure on the model and dumps all the intermediate
        and final results into the output logdir. It also stores the final
        model weights.

        Training runs in one of two regimes. With `training_options.epochs`
        set it runs for that many epochs and ends with a `final` snapshot.
        With `training_options.early_stopping_patience` set it runs until SER
        on `early_stopping_dataset` stops improving, and ends with an `ebest`
        snapshot of the best epoch, whose weights the model is left holding.

        :param shuffled_train_dataset: The dataset that should be used for training,
            wrapped in a shuffled view.
        :param training_options: Parameters of the training process.
        :param metrics: Which metrics the periodic evaluations report, both
            into the logdir and into tensorboard. Defaults to SER alone.
        :param logdir_path: Path to the directory where TensorBoard output
            will be logged as well as evaluation results and intermediate
            and final weights of the model.
        :param early_stopping_dataset: The dataset SER is measured on after
            every epoch to decide when to stop. Required exactly when the
            training options ask for early stopping. It may or may not be
            one of the dev datasets.
        """
        if training_options.is_early_stopping != (early_stopping_dataset is not None):
            raise ValueError(
                "An early stopping dataset is required exactly when "
                + "the training options set an early stopping patience."
            )

        # prepare the training dataset
        train_tf_dataset = construct_tf_dataset(
            shuffled_view=shuffled_train_dataset,
            architecture_options=self.architecture_options,
            token_map=self.token_map,
            training_or_inference_options=training_options,
        )

        # count training batches
        training_batch_count: int = len(train_tf_dataset)

        # prepare tensorboard
        tb_callback = tf.keras.callbacks.TensorBoard(str(logdir_path))

        # store training options in the logdir
        training_options.write_to_yaml_file(logdir_path / "training_options.yaml")

        metrics = metrics if metrics is not None else DEFAULT_METRICS

        early_stopping: EarlyStopping | None = None
        if training_options.early_stopping_patience is not None:
            early_stopping = EarlyStopping(training_options.early_stopping_patience)

        def snapshot_path(snapshot_name: str) -> Path:
            return logdir_path / "snapshots" / f"{snapshot_name}.model"

        # epochs whose snapshots currently exist, for the early stopping
        # garbage collection to choose from
        stored_epochs: set[int] = set()

        # define the evaluation callback
        # https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/callbacks/Callback#on_epoch_end
        zeus = self

        class EvaluationCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs=None):
                nonlocal zeus
                nonlocal inference_options_for_evaluation, logdir_path
                nonlocal training_options, dev_datasets, test_datasets, metrics
                nonlocal early_stopping, early_stopping_dataset, stored_epochs
                epoch_number = epoch + 1
                is_last_fixed_epoch = epoch_number == training_options.epochs
                is_tick = is_last_fixed_epoch or is_evaluation_tick(
                    epoch_number,
                    training_options.evaluation_from,
                    training_options.evaluation_each,
                )

                # Under early stopping any epoch may turn out to be the best,
                # so every one is stored and the surplus collected below.
                if is_tick or early_stopping is not None:
                    zeus.store(
                        snapshot_path(f"e{epoch_number}"),
                        musibot_model_version=zeus.snapshot_version(f"e{epoch_number}"),
                    )
                    stored_epochs.add(epoch_number)

                if early_stopping is not None:
                    assert early_stopping_dataset is not None
                    _, computed = zeus.evaluate(
                        dataset=early_stopping_dataset,
                        inference_options=inference_options_for_evaluation,
                        with_progress_bar=False,
                        metrics=[SER],
                    )
                    ser = computed[SER.name]
                    logs[f"early_stopping_{SER.name}"] = ser
                    improved = early_stopping.observe(epoch_number, ser)
                    print(
                        f"[Zeus]: Early stopping {SER.name} of epoch {epoch_number}: "
                        + f"{ser:.4f}"
                        + (" (new minimum)" if improved else "")
                        + f", best is epoch {early_stopping.best_epoch}."
                    )
                    write_early_stopping_record(
                        logdir_path / "early_stopping.yaml",
                        early_stopping,
                        early_stopping_dataset.name,
                    )

                if is_tick:
                    datasets_for_evaluation = dev_datasets
                    if is_last_fixed_epoch:
                        datasets_for_evaluation = dev_datasets + test_datasets

                    for dataset in datasets_for_evaluation:
                        # run evaluation and write predictions and metrics to logdir
                        evaluation_name = f"e{epoch_number}-{dataset.name}"
                        print("[Zeus]: Running evaluation", evaluation_name, "...")
                        _, computed = zeus.evaluate(
                            dataset=dataset,
                            inference_options=inference_options_for_evaluation,
                            with_progress_bar=False,
                            metrics=metrics,
                            write_predictions_to=logdir_path
                            / "evaluation"
                            / f"{evaluation_name}.lmx",
                            write_metrics_to=logdir_path / "evaluation" / f"{evaluation_name}.yaml",
                        )
                        # and write metrics to tensorboard
                        for metric_name, value in computed.items():
                            logs[f"{dataset.name}_{metric_name}"] = value

                    print(f"[Zeus]: Evaluation of epoch {epoch_number} done.")

                if early_stopping is not None:
                    for collected in snapshots_to_collect(
                        stored_epochs,
                        epoch_number,
                        early_stopping,
                        training_options.evaluation_from,
                        training_options.evaluation_each,
                    ):
                        shutil.rmtree(snapshot_path(f"e{collected}"))
                        stored_epochs.discard(collected)

                    if early_stopping.should_stop(epoch_number):
                        print(
                            f"[Zeus]: No new minimum for {early_stopping.patience} "
                            + f"epochs, stopping after epoch {epoch_number}."
                        )
                        self.model.stop_training = True

        # run training
        self.model.prepare_for_training(
            training_options=training_options, training_batch_count=training_batch_count
        )
        self.model.fit(
            train_tf_dataset,
            epochs=(
                training_options.epochs
                if training_options.epochs is not None
                else EARLY_STOPPING_EPOCH_LIMIT
            ),
            callbacks=[EvaluationCallback(), tb_callback],
            verbose=1,
        )

        if early_stopping is None:
            # store the final weights
            self.store(
                snapshot_path("final"),
                musibot_model_version=self.snapshot_version("final"),
            )
            return

        # Return to the best epoch, so that both the stored snapshot and the
        # model left in hand are the result of the run, not its overshoot.
        best_epoch = early_stopping.best_epoch
        assert best_epoch is not None
        self.load_weights(snapshot_path(f"e{best_epoch}"))
        print(f"[Zeus]: Restored the best epoch, {best_epoch}.")
        self.store(
            snapshot_path("ebest"),
            musibot_model_version=self.snapshot_version("ebest"),
        )

        for dataset in dev_datasets + test_datasets:
            evaluation_name = f"ebest-{dataset.name}"
            print("[Zeus]: Running evaluation", evaluation_name, "...")
            self.evaluate(
                dataset=dataset,
                inference_options=inference_options_for_evaluation,
                with_progress_bar=False,
                metrics=metrics,
                write_predictions_to=logdir_path / "evaluation" / f"{evaluation_name}.lmx",
                write_metrics_to=logdir_path / "evaluation" / f"{evaluation_name}.yaml",
            )

    def evaluate(
        self,
        dataset: ZeusDataset,
        inference_options: InferenceOptions,
        with_progress_bar: bool,
        metrics: list[Metric] | None = None,
        write_predictions_to: Path | None = None,
        write_metrics_to: Path | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Evaluate model on a given dataset.

        Returns LMX predictions (concatenated tokens) for all samples
        in the given dataset and then a dictionary of computed metrics
        on those predictions.

        It can also write both predictions and metrics to files if
        their paths are provided.

        :param metrics: Which metrics to compute, from `zeus.evaluation`.
            Defaults to SER alone.
        """
        metrics = metrics if metrics is not None else DEFAULT_METRICS
        # prepare the dataset
        tf_dataset = construct_tf_dataset(
            shuffled_view=ShuffledView.create_unshuffled_for(dataset),
            architecture_options=self.architecture_options,
            token_map=self.token_map,
            training_or_inference_options=inference_options,
        )

        # run model inference
        self.model.prepare_for_inference(inference_options)
        predicted_token_indexes = self.model.predict(
            tf_dataset, verbose=1 if with_progress_bar else 0
        )

        # decode to LMX
        predicted_lmx_samples: list[str] = [
            self.token_map.indices_to_lmx(list(sample_prediction.numpy()))
            for sample_prediction in predicted_token_indexes
        ]

        # compute metrics
        gold_lmx_samples = [sample.lmx for sample in dataset.samples]

        if with_progress_bar:
            print("Computing metrics...")

        computed_metrics = compute_metrics(metrics, gold_lmx_samples, predicted_lmx_samples)

        if with_progress_bar:
            print("Done. Metrics:", computed_metrics)

        # write predictions to file
        if write_predictions_to is not None:
            write_predictions_to.parent.mkdir(parents=True, exist_ok=True)
            write_predictions_to.write_text("\n".join(predicted_lmx_samples))

        # write metrics to file
        if write_metrics_to is not None:
            write_metrics_to.parent.mkdir(parents=True, exist_ok=True)
            write_metrics_to.write_text(
                "\n".join([f"{metric}: {value:.3f}" for metric, value in computed_metrics.items()])
            )

        return (predicted_lmx_samples, computed_metrics)

    def predict(
        self,
        images: list[bytes],
        inference_options: InferenceOptions,
        with_progress_bar: bool = False,
    ) -> list[str]:
        """Read music notation off the given images.

        This is inference proper: images in, transcriptions out, with no gold
        data anywhere. `evaluate` is the same forward pass with a dataset and a
        score attached; this is what you want when you have a scan and a
        question.

        The images are processed in batches of `inference_options.batch_size`,
        which is what makes this much faster than one call per image — a batch
        fills a single forward pass.

        :param images: Encoded image files, PNG or JPEG, one per staff or
            grandstaff. Each should be a single system: Zeus reads one staff at
            a time and will transcribe a whole page as if it were one.
        :param inference_options: Batch size, image transformations, and the
            limits on predicted length and image width.
        :param with_progress_bar: Show TensorFlow's progress bar.
        :returns: One LMX string per image, in the order the images were given.
            Use `zeus.musicxml.lmx_to_musicxml` to turn one into MusicXML.
        """
        if len(images) == 0:
            # tf.data cannot build a dataset with no elements to infer shapes
            # from, and there is nothing to predict anyway.
            return []

        tf_dataset = construct_tf_dataset_for_images(
            images=images,
            architecture_options=self.architecture_options,
            inference_options=inference_options,
        )

        self.model.prepare_for_inference(inference_options)
        predicted_token_indexes = self.model.predict(
            tf_dataset, verbose=1 if with_progress_bar else 0
        )

        predictions = [
            self.token_map.indices_to_lmx(list(sample_prediction.numpy()))
            for sample_prediction in predicted_token_indexes
        ]

        assert len(predictions) == len(images), (
            f"The model returned {len(predictions)} predictions for {len(images)} images; "
            "the caller matches them up by position, so this must not happen."
        )

        return predictions
