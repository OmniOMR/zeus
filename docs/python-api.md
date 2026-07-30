# Python API

Zeus is a python package as well as a command line tool. If your project runs on python 3.10, you can drive the model directly instead of shelling out to the CLI — the CLI is a thin layer over exactly these calls.

If your project cannot be on python 3.10, use [the CLI](using-the-cli.md) instead. The constraint is TensorFlow 2.12, which has no wheels for anything newer, and it is not one Zeus can lift on its own.


## The public surface

Everything a library user needs is importable from the package root:

```py
from zeus import Zeus, InferenceOptions, TrainingOptions, ZeusDataset
```

| Name | What it is |
| --- | --- |
| `Zeus` | The model: weights and architecture. Loaded from a snapshot or created fresh, then trained, evaluated or run for inference. |
| `ArchitectureOptions` | The sizes, dimensions and layer counts that define one architecture, such as `grand24` or `solo26`. |
| `ModelOptions` | What a trained model reads, as opposed to how it computes — today, which [page subdivisions](model-snapshots.md#architecture-options-versus-model-options) it accepts. |
| `TrainingOptions` | What a training run does: epochs, batch size, learning rate, augmentations. |
| `InferenceOptions` | What an inference run does: batch size, image transformations, length and width limits. |
| `TokenMap` | The mapping between model output indices and LMX tokens. Part of a snapshot, and inseparable from its weights. |
| `ZeusDataset` | A dataset split loaded in memory, as a list of samples. |
| `ZeusDatasetSample` | One sample: an image and its LMX. |
| `ShuffledView` | A shuffled view over a `ZeusDataset`, which is what training consumes. |

These names are the API. The module paths beneath them (`zeus.model.zeus`, `zeus.data.zeus_dataset`, …) are internal arrangement and may move; anything re-exported above will not move without a version bump.


## Importing is cheap; using is not

`import zeus` imports nothing at all. Each name above is resolved the first time it is touched, through a module-level `__getattr__`:

```py
import zeus  # instant — no TensorFlow
from zeus import Zeus  # this is what imports TensorFlow, and it takes seconds
```

That is worth knowing in two situations. If you are measuring startup, the cost lands at first *use*, not at the import statement. And if you are writing something that only wants a dataclass — reading `InferenceOptions` defaults, say — you never pay for TensorFlow at all, because nothing in the options, the token map or the dataset classes needs it.

Type checkers and editors see the real definitions, so `from zeus import Zeus` type-checks and autocompletes normally.


## Loading a snapshot

A [snapshot](model-snapshots.md) is a folder with the `.model` suffix holding the weights, the architecture options and the token map. Loading one gives you a model ready to use:

```py
from pathlib import Path

from zeus import Zeus

model = Zeus.load(Path("models/zeus-olimpic-1.0-2024-02-12.model"))
```

`Zeus.load` reads the architecture options, the model options and the token map out of the folder, builds the model, runs one dummy inference to materialize the decoder's weights, and only then loads the weights file. Snapshots from 2024 are detected and loaded correctly too.

Storing works the same way round:

```py
model.store(Path("models/my-model.model"), overwrite=False)
```


## Evaluating against a dataset

```py
from pathlib import Path

from zeus import InferenceOptions, Zeus, ZeusDataset

dataset = ZeusDataset.load_from_pickle_file(Path("datasets/omniomr/samples.test.pickle"))
model = Zeus.load(Path("models/solo26.model"))

predictions, metrics = model.evaluate(
    dataset=dataset,
    inference_options=InferenceOptions(batch_size=64),
    with_progress_bar=True,
)

print(metrics)  # {'SER': 4.12}
```

`predictions` is one LMX string per sample, in the dataset's own order. Passing `write_predictions_to` and `write_metrics_to` also writes them to disk, which is what `zeus evaluate` does.

`metrics=` chooses what is computed, defaulting to SER alone:

```py
from zeus.evaluation import SER, SER_PITCH_ONLY

predictions, metrics = model.evaluate(
    dataset=dataset,
    inference_options=InferenceOptions(batch_size=64),
    with_progress_bar=True,
    metrics=[SER, SER_PITCH_ONLY],
)
```

`Zeus.train` takes the same argument, deciding what its periodic evaluations report. See [Evaluation metrics](evaluation-metrics.md) for what each one measures, and for how to add another.

See [Zeus dataset format and pickling](zeus-dataset-format-and-pickling.md) for where those pickles come from.


## Training

```py
from pathlib import Path

from zeus import (
    ArchitectureOptions,
    InferenceOptions,
    ModelOptions,
    ShuffledView,
    TokenMap,
    TrainingOptions,
    Zeus,
    ZeusDataset,
)

train_dataset = ZeusDataset.load_from_pickle_file(Path("datasets/omniomr/samples.train.pickle"))

model = Zeus(
    architecture_options=ArchitectureOptions.from_well_known("solo26"),
    token_map=TokenMap.create_from_dataset(train_dataset.samples),
    model_options=ModelOptions(input_subdivisions=["Staves"]),
)

model.train(
    shuffled_train_dataset=ShuffledView.create_random_for(train_dataset, seed=42),
    dev_datasets=[],
    test_datasets=[],
    training_options=TrainingOptions(
        epochs=50,
        evaluation_from=1,
        evaluation_each=1,
        is_finetuning=False,
    ),
    inference_options_for_evaluation=InferenceOptions(batch_size=64),
    logdir_path=Path("logs/my-experiment"),
)
```

Two things are easy to get wrong here. The token map must come from the training data when training from scratch, and must come from the snapshot when fine-tuning — `Zeus.load` handles the second case for you, and a model cannot learn tokens its map does not contain. And `is_finetuning` is not merely a label: it decides whether an unknown token in the training data is an error or is quietly encoded as `<unk>`.

See [Training Zeus](training-zeus.md) for what the options mean and what lands in the logdir.


## Inference on images

`evaluate` needs a dataset, and a dataset needs gold LMX for every sample — reasonable when measuring a model, impossible when merely running one. `predict` takes encoded images and nothing else:

```py
from pathlib import Path

from zeus import InferenceOptions, Zeus
from zeus.musicxml.lmx_to_musicxml import lmx_to_musicxml

model = Zeus.load(Path("models/solo26.model"))

images = [path.read_bytes() for path in Path("staves").glob("*.jpg")]
predictions = model.predict(images, InferenceOptions(batch_size=16))

for lmx in predictions:
    print(lmx_to_musicxml(lmx))
```

Pass every image you have in one call rather than looping: a batch fills a single forward pass, and that is most of what makes inference fast. `predictions[i]` corresponds to `images[i]`.

Each image should be a single staff or grandstaff. Zeus reads one system at a time, and handed a whole page it will transcribe the page as though it were one staff.


### From LMX to MusicXML

The model predicts LMX, because that is what an image-to-sequence model can emit. `lmx_to_musicxml` is the last step:

```py
from zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml

try:
    musicxml = lmx_to_musicxml(lmx)
except LmxDecodingError as error:
    print("this staff could not be read:", error)
```

Catching that is worth doing whenever you are transcribing more than one image. A model's output is a prediction, not a promise: nothing constrains it to be well-formed LMX, and a confused model does emit sequences the decoder cannot make sense of. `LmxDecodingError` is raised for all of them, so one unreadable staff costs that staff rather than the run.


## Image preprocessing

Every path — training, evaluation and prediction — normalizes images through one function, `zeus.model.preprocess_image`, so that a model cannot be trained on images prepared one way and then fed images prepared another.

Two properties of it are worth knowing, because they shape what the model sees:

- **The height is exact.** Images are scaled to `architecture_options.height` — 96 for `solo26`, 192 for `grand24` — and never to anything else. The encoder declares that height statically.
- **The width is capped, by squashing.** Past `InferenceOptions.max_image_width` (1500 by default) the image is compressed horizontally rather than scaled down as a whole. Aspect ratio, exact height and a bounded width cannot all three hold, and aspect ratio is the one to give up: vertical resolution is where pitch lives, while horizontal compression only crowds the notes together. Set `max_image_width=None` to lift the cap and let wide staves be as wide as they are.
