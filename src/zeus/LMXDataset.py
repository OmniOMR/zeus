from pathlib import Path
from .RawDatasetSample import RawDatasetSample
import tensorflow as tf
import re
import numpy as np
import random
from .ArchitectureOptions import ArchitectureOptions
from .TokenMap import TokenMap
from typing import Iterable
from .TrainingOptions import TrainingOptions
from .InferenceOptions import InferenceOptions


class LMXDataset:
    """Represents a dataset of images and their LMX annotations"""
    
    def __init__(self, name: str, samples: list[RawDatasetSample]) -> None:
        self.name = name
        """Human-readable name of the dataset, used in logs, must be path-safe"""

        self.samples = samples
        """Inidividual samples of the dataset, ordered in the same
        way as the samples txt file"""

        self.shuffled_view: list[int] = list(range(len(self.samples)))
        """One fixed shuffled view at the dataset that should be used
        as the sample order during training"""

        # actually shuffle the shuffled view
        random.Random(42).shuffle(self.shuffled_view)

    @staticmethod
    def from_pickle_file(pickle_path: Path) -> "LMXDataset":
        """Loads a dataset from its pickled representation"""
        samples = RawDatasetSample.load_samples(pickle_path)
        name = pickle_path.as_posix()
        if name.startswith("datasets/"):
            name = name[len("datasets/"):]
        name = name.replace("/", "_")
        return LMXDataset(
            name=name,
            samples=samples,
        )
    
    @staticmethod
    def combine_multiple(datasets: list["LMXDataset"]) -> "LMXDataset":
        """Combines multiple LMX datasets into one"""
        assert len(datasets) > 0

        # only one
        if len(datasets) == 1:
            return datasets[0]
        
        # combine
        name = ""
        samples: list[RawDatasetSample] = []
        for dataset in datasets:
            samples += dataset.samples
            if name != "":
                name += "-and-"
            name += dataset.name
        return LMXDataset(
            name=name,
            samples=samples,
        )
    
    def print_statistics(self):
        """Prints dataset statistics into the console"""
        avg_len = np.mean(
            [len(sample.lmx.split()) for sample in self.samples]
        )
        print(
            f"Loaded dataset {self.name}, {len(self.samples)} " +
            f"examples, {avg_len:.2f} avg length."
        )

    def construct_tf_dataset(
            self,
            architecture_options: ArchitectureOptions,
            token_map: TokenMap,
            training_or_inference_options: TrainingOptions | InferenceOptions,
    ) -> tf.data.Dataset:
        """
        Returns a tf.data.Dataset representation of the data for either
        training or inference, based on the given options object.

        When training, the data is shuffled and augmented.

        When running inference, images may be transformed before
        being fed into the model (i.e. binarized).
        
        :param architecture_options: Used to get the desired image height.
        :param token_map: Map used to convert LMX tokens to feature indexes
            for the model output layer.
        :param training_or_inference_options: Options to either prepare the
            data for training or for inference.
        """

        # parse out needed values from options
        options = training_or_inference_options
        if isinstance(options, TrainingOptions):
            is_training = True
            is_finetuning = options.is_finetuning
            transformations: list[str] = []
            augmentations: str = options.augmentations
            batch_size: int = options.batch_size
            seed: int = options.seed
            max_image_width: int | None = options.max_image_width
        elif isinstance(options, InferenceOptions):
            is_training = False
            is_finetuning = False
            transformations = options.transformations
            augmentations = "" # not used
            batch_size = options.batch_size
            seed = 0 # not used
            max_image_width = options.max_image_width

        def generator() -> Iterable[tuple[bytes, np.ndarray]]:
            """
            Emits pairs of (image, lmx token indices), where the image is kept
            in its original binary representation to conserve RAM space.
            Images are decoded only as they are needed for the next batch.
            LMX is however decoded into indexes right away.
            """
            nonlocal is_training, token_map
            for i in range(len(self.samples)):
                sample = self.samples[self.shuffled_view[i]] \
                    if is_training else self.samples[i]
                yield (
                    sample.image,
                    np.array([
                        token_map.token_to_index(
                            token,
                            allow_unknown_tokens=(not is_training or is_finetuning)
                        )
                        for token in sample.lmx.split()
                    ], dtype=np.int32)
                )

        def prepare_example(image: bytes, token_indexes: np.ndarray) -> tuple[tf.Tensor, np.ndarray]:
            nonlocal architecture_options, transformations, max_image_width
            image = tf.image.convert_image_dtype(tf.image.decode_image(image, channels=1, expand_animations=False), tf.float32)
            for transformation, *parameters in map(lambda part: part.split(":"), transformations):
                if transformation == "threshold":
                    l, r, *rest = parameters
                    l, r, smooth = float(l), float(r), rest.count("smooth")
                    if not smooth:
                        image = tf.cast(image >= l, tf.float32) * tf.cast(image <= r, tf.float32) * image + tf.cast(image > r, tf.float32)
                    else:
                        image = tf.clip_by_value((image - l) / (r - l), 0., 1.)
                elif transformation:
                    raise ValueError(f"The transformation '{transformation}' is unknown.")
            image = tf.image.resize(
                image,
                size=[
                    architecture_options.height,
                    max_image_width or tf.int32.max
                ],
                preserve_aspect_ratio=True,
                antialias=True
            )
            return image, token_indexes
        
        if is_training:
            # Prepare augmentation operations, reused in between batches
            rng = tf.random.Generator.from_seed(seed)
            if match := re.search(r"rotate:([\d.]+)", augmentations):
                rng_rotation = tf.keras.layers.RandomRotation(
                    float(match.group(1)) / 360, fill_mode="constant", interpolation="bilinear", seed=seed, fill_value=1.0)

        def augment(image, tags):
            nonlocal rng, rng_rotation, augmentations, architecture_options
            for augmentation, *parameters in map(lambda part: part.split(":"), augmentations.split(",")):
                if rng.uniform([], 0, 1) >= 0.5:
                    continue
                if augmentation == "h":
                    a = int(parameters[0])
                    image = tf.pad(image, [[0, 0], [a, 0], [0, 0]], constant_values=1.0)[:, rng.uniform([], 0, 2 * a + 1, dtype=tf.int32):]
                elif augmentation == "v":
                    a = int(parameters[0])
                    image = tf.pad(image, [[a, a], [0, 0], [0, 0]], constant_values=1.0)[rng.uniform([], 0, 2 * a + 1, dtype=tf.int32):]
                    image = image[:architecture_options.height]
                elif augmentation == "rotate":
                    image = rng_rotation(image, training=True)
                elif augmentation == "b":
                    l, u = map(float, parameters[:2])
                    image = tf.clip_by_value(tf.image.adjust_brightness(image, rng.uniform([], l, u)), 0., 1.)
                elif augmentation == "c":
                    l, u = map(float, parameters[:2])
                    image = tf.clip_by_value(tf.image.adjust_contrast(image, 2 ** rng.uniform([], l, u)), 0., 1.)
                elif augmentation == "n":
                    p = float(parameters[0])
                    mask = tf.cast(rng.uniform(tf.shape(image), 0, 1) >= rng.uniform([], 0, p), tf.float32)
                    image = mask * image + (1 - mask) * (1 - image)
                elif augmentation == "en3":
                    p = float(parameters[0])
                    mask = tf.nn.avg_pool2d(image[tf.newaxis], 3, 1, padding="SAME")[0]
                    mask = tf.cast((mask <= 0.1) | (mask >= 0.9), tf.float32)
                    mask = mask + (1 - mask) * tf.cast(rng.uniform(tf.shape(mask), 0, 1) >= rng.uniform([], 0, p), tf.float32)
                    image = mask * image + (1 - mask) * (1 - image)
                elif augmentation == "de":
                    d = rng.uniform([], -np.pi / 2, np.pi / 2)
                    x, y = tf.cos(d), 0.5 * tf.sin(d)
                    moved = tf.raw_ops.ImageProjectiveTransformV3(
                        images=image[tf.newaxis], transforms=[[1., 0., x, 0., 1., y, 0., 0.]],
                        output_shape=tf.shape(image)[:2], fill_value=1.0, interpolation="BILINEAR")[0]
                    if rng.uniform([], 0, 1) >= 0.5:
                        image = tf.math.maximum(image, moved)
                    else:
                        image = tf.clip_by_value(image + moved - 1, 0., 1.)
                elif augmentation:
                    raise ValueError(f"The augmentation '{augmentation}' is unknown.")
            return image, tags
        
        # === The dataset pipeline ===

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        dataset = dataset.cache()
        dataset = dataset.apply(
            tf.data.experimental.assert_cardinality(
                expected_cardinality=sum(1 for _ in dataset)
            )
        )
        if is_training:
            dataset = dataset.shuffle(5_000, seed=seed)
        dataset = dataset.map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training and augmentations:
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.ragged_batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
