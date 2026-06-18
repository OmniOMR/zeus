from .ArchitectureOptions import ArchitectureOptions
from pathlib import Path
import random
import shutil
import tensorflow as tf
import contextlib
from .KerasModel import KerasModel
from .TokenMap import TokenMap
from ..data.ZeusDataset import ZeusDataset
from ..data.ShuffledView import ShuffledView
from .TrainingOptions import TrainingOptions
from .InferenceOptions import InferenceOptions
from .construct_tf_dataset import construct_tf_dataset
from ..evaluation.ser_metric import ser_metric


class Zeus:
    """
    The Zeus model - weigths and architecture.
    Can be trained or used for inference.
    """

    def __init__(
            self,
            architecture_options: ArchitectureOptions,
            token_map: TokenMap
    ):
        """Creates a fresh, initialized and untrained model instance."""

        self.architecture_options = architecture_options
        """Defines concrete sizes, dimensions, and layer counts"""

        self.token_map = token_map
        """Defines the mapping between model outputs and LMX tokens"""

        strategy_scope = tf.distribute.MirroredStrategy().scope() \
            if len(tf.config.list_physical_devices("GPU")) > 1 \
            else contextlib.nullcontext()
        with strategy_scope:
            model = KerasModel(
                architecture_options=architecture_options,
                token_map=token_map
            )
        
        self.model: KerasModel = model
        """The Keras model containing all the low-level tensorflow stuff"""

    @staticmethod
    def load(model_folder_path: Path) -> "Zeus":
        """Loads a model from its folder"""
        architecture_options = ArchitectureOptions.from_model_folder(
            model_folder_path
        )
        token_map = TokenMap.load_from_model_folder(
            model_folder_path
        )
        zeus = Zeus(
            architecture_options=architecture_options,
            token_map=token_map
        )

        # run one dummy decoder inference to fully build the model
        # or something like that, so that weights can be loaded
        zeus.model.decoder_inference(
            encoded=zeus.model.encoder(
                tf.RaggedTensor.from_tensor(
                    tf.ones(
                        [1, zeus.architecture_options.height, 128, 1],
                        dtype=tf.float32
                    ),
                    ragged_rank=2
                )
            ),
            max_length=1,
        )

        # keras Layer property, prevents Keras
        # from calling "build" on invocation
        # (and thus overwriting loaded weights)
        # https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/layers/Layer
        # https://github.com/keras-team/keras/blob/v2.12.0/keras/engine/base_layer.py#L364
        zeus.model.built = True

        # load weights
        zeus.model.load_weights(
            str(model_folder_path / "weights.h5")
        )

        print("[Zeus]: Loaded model", model_folder_path)
        
        return zeus
    
    def store(self, model_folder_path: Path, overwrite=False):
        """Stores the model weights and parameters into a folder"""
        if model_folder_path.exists():
            if overwrite:
                shutil.rmtree(model_folder_path)
            else:
                raise Exception("Cannot store model, folder already exists.")
        
        # create the target folder
        model_folder_path.mkdir(parents=True, exist_ok=True)

        # model weights
        self.model.save_weights(str(model_folder_path / "weights.h5"))

        # architecture options
        self.architecture_options.write_to_model_folder(model_folder_path)

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
    ):
        """
        Runs a training procedure on the model and dumps all the intermediate
        and final results into the output logdir. It also stores the final
        model weights.

        :param shuffled_train_dataset: The dataset that should be used for training,
            wrapped in a shuffled view.
        :param training_options: Parameters of the training process.
        :param logdir_path: Path to the directory where TensorBoard output
            will be logged as well as evaluation results and intermediate
            and final weights of the model.
        """
        # prepare the training dataset
        train_tf_dataset = construct_tf_dataset(
            shuffled_view=shuffled_train_dataset,
            architecture_options=self.architecture_options,
            token_map=self.token_map,
            training_or_inference_options=training_options
        )

        # count training batches
        training_batch_count: int = len(train_tf_dataset)

        # prepare tensorboard
        tb_callback = tf.keras.callbacks.TensorBoard(str(logdir_path))

        # store training options in the logdir
        training_options.write_to_yaml_file(
            logdir_path / "training_options.yaml"
        )

        # define the evaluation callback
        # https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/callbacks/Callback#on_epoch_end
        zeus = self
        class EvaluationCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs=None):
                nonlocal zeus
                nonlocal inference_options_for_evaluation, logdir_path
                nonlocal training_options, dev_datasets, test_datasets
                if epoch + 1 < training_options.epochs and (
                    epoch + 1 < training_options.evaluation_from or
                        (epoch + 1) % training_options.evaluation_each != 0
                ): return

                # store model weights to logdir
                zeus.store(logdir_path / "snapshots" / f"e{epoch + 1}.model")

                datasets_for_evaluation = dev_datasets
                if epoch + 1 == training_options.epochs:
                    datasets_for_evaluation += test_datasets

                for dataset in datasets_for_evaluation:
                    # run evaluation and write predictions and metrics to logdir
                    evaluation_name = f"e{epoch + 1}-{dataset.name}"
                    print("[Zeus]: Running evaluation", evaluation_name, "...")
                    _, metrics = zeus.evaluate(
                        dataset=dataset,
                        inference_options=inference_options_for_evaluation,
                        with_progress_bar=False,
                        write_predictions_to=logdir_path / "evaluation" / f"{evaluation_name}.lmx",
                        write_metrics_to=logdir_path / "evaluation" / f"{evaluation_name}.yaml",
                    )
                    # and write metrics to tensorboard
                    for metric, value in metrics.items():
                        logs[f"{dataset.name}_{metric}"] = value
                
                print(f"[Zeus]: Evaluation of epoch {epoch + 1} done.")

        # run training
        self.model.prepare_for_training(
            training_options=training_options,
            training_batch_count=training_batch_count
        )
        self.model.fit(
            train_tf_dataset,
            epochs=training_options.epochs,
            callbacks=[EvaluationCallback(), tb_callback],
            verbose=1
        )

        # store the final weights
        self.store(logdir_path / "snapshots" / "final.model")

    def visualize_training_data(
            self,
            shuffled_train_dataset: ShuffledView,
            training_options: TrainingOptions,
            output_folder_path: Path,
            sample_count=100,
    ):
        """
        Creates visualisation of training data.

        Writes images and an index.html file into the given folder
        (creates the folder if missing).
        """
        # prepare the training dataset
        train_tf_dataset = construct_tf_dataset(
            shuffled_view=shuffled_train_dataset,
            architecture_options=self.architecture_options,
            token_map=self.token_map,
            training_or_inference_options=training_options
        )

        # count training batches
        training_batch_count: int = len(train_tf_dataset)
        print("There are", training_batch_count, "batches in the dataset")

        # prepare the visualization folder
        output_folder_path.mkdir(parents=True, exist_ok=True)
        images_folder_path = output_folder_path / "images"
        images_folder_path.mkdir(exist_ok=True)
        html_file_path = output_folder_path / "index.html"
        
        # dump images and HTML
        html = f"<html><body><h1>{shuffled_train_dataset.dataset.name}</h1>"
        sample_index = 0
        batch_index = 0
        for batch_images, batch_annotations in train_tf_dataset:
            assert batch_images.shape[0] == batch_annotations.shape[0]
            html += f"<h2>Batch {batch_index}</h2>"
            for i in range(batch_images.shape[0]):
                png_bytes = tf.image.encode_png(
                    tf.cast(batch_images[i].to_tensor() * 255, tf.uint8)
                ).numpy()
                (images_folder_path / f"{sample_index}.png").write_bytes(
                    png_bytes
                )
                lmx = self.token_map.indices_to_lmx(
                    batch_annotations[i].numpy()
                )
                html += "<div>"
                html += f'<img src="images/{sample_index}.png">'
                html += f'<p>LMX: <code>{lmx}</code></p>'
                html += "</div>"
                
                sample_index += 1

                if sample_index >= sample_count:
                    break
            
            batch_index += 1

            if sample_index >= sample_count:
                    break

        # complete the html file and write it
        html += "</body></html>"
        html_file_path.write_text(html)

        print("Visualisation has been written to", output_folder_path)
    
    def visualize_predictions(
            self,
            title: str,
            dataset: ZeusDataset,
            predictions_lmx: list[str],
            output_html_path: Path,
            sample_count=100,
    ):
        """
        Creates a visualization of predicted LMX values.
        Combines the LMX with images taken from a dataset pickle
        and produces an html file that showcases a random set
        of samples, ordered by SER from the best to worst.
        Median error is in the middle of the html file.
        """
        assert output_html_path.suffix == ".html"
        assert len(dataset.samples) == len(predictions_lmx), \
            "Given dataset has different number of samples than " + \
            "the precitions LMX file. Did you provide the correct dataset?"
        
        # compute sample indices we'll take
        # (permute and possibly sub-sample)
        sample_indices = list(range(len(predictions_lmx)))
        random.Random(42).shuffle(sample_indices)
        if len(sample_indices) > sample_count:
            sample_indices = sample_indices[:sample_count]

        # prepare the image folder
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        images_folder_path = output_html_path.parent / (
            output_html_path.stem + "-imgs"
        )
        images_folder_path.mkdir(exist_ok=True)

        # prepare items for the html
        # (sample_index, img, gold, pred, ser)
        items: list[tuple[int, bytes, str, str, float]] = []
        for sample_index in sample_indices:
            sample = dataset.samples[sample_index]
            gold_lmx: str = sample.lmx
            pred_lmx: str = predictions_lmx[sample_index]
            ser: float = ser_metric(
                gold=[gold_lmx],
                pred=[pred_lmx]
            )["SER"]
            items.append(
                (sample_index, sample.image, gold_lmx, pred_lmx, ser)
            )

        # sort items by SER
        items.sort(key=lambda item: item[4])

        # dump items to HTML
        html = f"<html><body><h1>{title} @ {dataset.name}</h1>"
        for sample_index, image, gold_lmx, pred_lmx, ser in items:
            (images_folder_path / f"{sample_index}.jpg").write_bytes(
                image
            )
            html += "<div>"
            html += f'<img src="{images_folder_path.name}/{sample_index}.jpg" height="192">'
            html += f'<p>SER: <strong>{ser:.2f}</strong></p>'
            html += f'<p>Gold LMX: <code>{gold_lmx}</code></p>'
            html += f'<p>Predicted LMX: <code>{pred_lmx}</code></p>'
            html += "</div>"
            
        # complete the html file and write it
        html += "</body></html>"
        output_html_path.write_text(html)

        print("Visualisation has been written to", output_html_path)
    
    def evaluate(
            self,
            dataset: ZeusDataset,
            inference_options: InferenceOptions,
            with_progress_bar: bool,
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
        """
        # prepare the dataset
        tf_dataset = construct_tf_dataset(
            shuffled_view=ShuffledView.create_unshuffled_for(dataset),
            architecture_options=self.architecture_options,
            token_map=self.token_map,
            training_or_inference_options=inference_options
        )

        # run model inference
        self.model.prepare_for_inference(inference_options)
        predicted_token_indexes = self.model.predict(
            tf_dataset,
            verbose=1 if with_progress_bar else 0
        )

        # decode to LMX
        predicted_lmx_samples: list[str] = [
            self.token_map.indices_to_lmx(list(sample_prediction.numpy()))
            for sample_prediction in predicted_token_indexes
        ]

        # compute metrics
        gold_lmx_samples = [sample.lmx for sample in dataset.samples]
        computed_metrics: dict[str, float] = {}

        if with_progress_bar:
            print("Computing metrics...")

        computed_metrics.update(
            ser_metric(gold_lmx_samples, predicted_lmx_samples)
        )

        if with_progress_bar:
            print("Done. Metrics:", computed_metrics)

        # write predictions to file
        if write_predictions_to is not None:
            write_predictions_to.parent.mkdir(parents=True, exist_ok=True)
            write_predictions_to.write_text("\n".join(predicted_lmx_samples))
        
        # write metrics to file
        if write_metrics_to is not None:
            write_metrics_to.parent.mkdir(parents=True, exist_ok=True)
            write_metrics_to.write_text("\n".join([
                f"{metric}: {value:.3f}"
                for metric, value in computed_metrics.items()
            ]))

        return (predicted_lmx_samples, computed_metrics)

    def predict(self, inference_options: InferenceOptions):
        """Run model inference on given images of music notation"""
        raise NotImplementedError
