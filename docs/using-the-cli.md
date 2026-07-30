# Using the CLI

Zeus installs one command, `zeus`, which is how you transcribe images, measure a model against a dataset, and train one. It is the whole of what Zeus offers to a project that cannot run on python 3.10 itself — everything below shells out, so nothing about your own environment has to change.

If your project *is* on python 3.10, the [Python API](python-api.md) does the same things in-process and lets you keep a loaded model between calls. The CLI is a thin layer over exactly those calls.

Every command explains itself with `--help`, and that is worth using: this page covers the three things people do most, not every argument.

```bash
zeus --help
zeus predict --help
```


## Installing

Zeus needs **python 3.10** and refuses to install on anything newer, because TensorFlow 2.12 has no wheels for 3.11 or later. Give it a virtual environment of its own:

```bash
python3.10 -m venv .venv
.venv/bin/pip install 'zeus @ git+https://github.com/OmniOMR/zeus.git@v1.0.0'
```

Replace the tag with `main` to follow development. Every commit builds as a distinct version, so a plain `pip install` of a newer commit replaces what is there — see [Versioning and releases](versioning-and-releases.md).

To work on Zeus itself rather than to use it, see [Development](development.md) instead.


## Getting a model snapshot

Zeus ships no weights. A [snapshot](model-snapshots.md) is a folder with the `.model` suffix, distributed as a `.tar.gz`; the README lists [the published ones](../README.md#existing-snapshots).

```bash
mkdir -p models && cd models
curl -LO https://github.com/ufal/olimpic-icdar24/releases/download/zeus-release/zeus-olimpic-1.0-2024-02-12.model.tar.gz
tar -xzf zeus-olimpic-1.0-2024-02-12.model.tar.gz
cd ..
```

You should end up with a *folder* named `models/zeus-olimpic-1.0-2024-02-12.model`, not a file. That folder path is what every command below takes as `--model-snapshot`.

Which images a snapshot expects is not a matter of taste — a grandstaff model reads two staves braced together, a solo-staff model reads one, and neither does anything sensible with the other's input. The 2024 snapshots above are all grandstaff models.


## Transcribing images

This is what most people came for. Give `zeus predict` one or more images and it writes a MusicXML file beside each:

```bash
zeus predict \
    --model-snapshot models/zeus-olimpic-1.0-2024-02-12.model \
    --quiet-tf \
    scans/grandstaff-1.jpg scans/grandstaff-2.jpg
```

```
Wrote scans/grandstaff-1.musicxml
Wrote scans/grandstaff-2.musicxml
```

`--output-dir out/transcriptions` sends them elsewhere instead, and `--lmx` writes the raw [LMX](musicxml-lmx-and-tokenization.md) token sequence beside each MusicXML file — the model's actual output, before it was turned into MusicXML. Reach for that when a transcription looks wrong, because a decoding problem is invisible in the MusicXML.

**Each image must be a single staff or grandstaff**, cropped and roughly deskewed. Zeus reads one system at a time; handed a whole page it will not complain, it will transcribe the page as though it were one staff and return nonsense. Finding and cropping the staves on a page is a different model's job — which is what [Musibot pipelines](musibot-model.md) exist to string together.


### One invocation, many images

Pass every image you have to a single command rather than calling `zeus predict` in a loop.

Loading the snapshot takes several seconds — TensorFlow's import, then building the network and reading the weights — and it happens once per invocation regardless of how many images follow. The transcription itself is batched, so images go through the model several at a time in one forward pass. A hundred images in one command is therefore nowhere near a hundred times the work of one; a hundred separate commands is a hundred times the *loading*.

`--batch-size` (16 by default) bounds how many enter one forward pass. Raise it if you have GPU memory to spare, lower it if you run out.

`--help` never loads TensorFlow, which is why it answers instantly while the real commands pause first.


### When a staff cannot be read

A model's output is a prediction, not a promise, and a hard image can produce a token sequence that is not valid LMX. That fails only its own image:

```
Failed on scans/staff-3.jpg: Could not decode the predicted LMX into MusicXML: …
Wrote scans/staff-4.musicxml

1 of 4 images could not be transcribed.
```

The command exits with status 1 if anything failed, so a script can notice, while everything that did work has still been written. With `--lmx` the failed image's tokens are on disk too, which is when you most want them.


## Evaluating a snapshot against a dataset

Measuring a model needs gold transcriptions, so this takes a [dataset pickle](zeus-dataset-format-and-pickling.md) rather than loose images:

```bash
zeus evaluate \
    --model-snapshot models/solo26.model \
    --dataset datasets/omniomr/samples.test.pickle \
    --output out/evaluation
```

That writes two files into the output folder:

- `predictions.lmx` — one predicted LMX string per sample, in the dataset's order.
- `metrics.yaml` — the symbol error rates.

```
SER: 4.123
SERnotuplets: 3.870
```

`SER` is the symbol error rate over LMX tokens: the edit distance between predicted and gold, as a percentage of gold length. `SERnotuplets` is the same with tuplet tokens excluded, since tuplets are both rare and easy to get wrong, and can dominate the first number.

Both are computed on the token sequence rather than on the music, so they are sensitive to how the notation was linearized — two transcriptions a musician would call equivalent can differ in LMX and be counted as error. Treat SER as a training signal rather than as a claim about musical correctness; see [Rough edges](rough-edges.md).

A number alone will not tell you *how* a model fails. Feed the predictions to [`zeus visualize-predictions`](visualizing-data-and-predictions.md) to see them beside their images, ordered worst-last.


## Training

Training is its own subject with its own page — see [Training Zeus](training-zeus.md) for what every argument means, how fine-tuning differs, and what lands in the log directory. The shape of it:

```bash
zeus train \
    --experiment solo26-omniomr \
    --architecture solo26 \
    --input-subdivisions Staves \
    --train datasets/omniomr/samples.train.pickle \
    --dev datasets/omniomr/samples.validation.pickle \
    --test datasets/omniomr/samples.test.pickle \
    --epochs 500 \
    --batch-size 64 \
    --quiet-tf
```

Two of those are not optional knobs and are worth knowing before you start. `--experiment` names the run, and becomes the model's name if it is ever deployed under Musibot. `--input-subdivisions` says what the model will be trained to read; it is stored in the snapshot, and it is what stops a grandstaff model being handed staves later.

Fine-tuning an existing snapshot uses `--model-snapshot` in place of `--architecture`, and inherits both of those from what it loads.

Getting data into the shape `--train` expects is a separate job: see [Converting MusiCorpus datasets to Zeus format](converting-musicorpus-datasets-to-zeus-format.md) and [Zeus dataset format and pickling](zeus-dataset-format-and-pickling.md).


## Where things are written

Several commands default to paths relative to the working directory, so Zeus is normally run from one place — the root of a project or of a checkout:

| Folder | Written by |
| --- | --- |
| `logs/` | `zeus train` — tensorboard, evaluations, and a snapshot per evaluated epoch |
| `out/` | `zeus evaluate` and the visualization commands, when `--output` is left alone |
| `datasets/`, `models/` | by convention, where the data and snapshots you fetch tend to live |

Nothing enforces that arrangement; it is what the defaults assume. Pass explicit paths and they are used instead.


## Running Zeus as a service

`zeus musibot` is not for interactive use. It runs Zeus as a [Musibot](https://github.com/OmniOMR/musibot) model, driven over pipes by a worker head, and started by hand it says so and exits. See [Running Zeus as a Musibot model](musibot-model.md).
